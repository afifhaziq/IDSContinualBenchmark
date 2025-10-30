# /// script
# requires-python = "==3.12.4"
# dependencies = []
# ///

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.scenarios.dataset_scenario import split_validation_class_balanced
from avalanche.benchmarks.scenarios.validation_scenario import benchmark_with_validation_stream
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, class_accuracy_metrics, StreamConfusionMatrix, forgetting_metrics, bwt_metrics
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.logging import InteractiveLogger, WandBLogger, TextLogger
import matplotlib.pyplot as plt
import random
import os
from preprocess import TabularDataset, AvalancheTabularDataset, SklearnStratifiedPerExpSplit
import yaml
from tab_transformer_pytorch import TabTransformer
import wandb
from utils import log_single_metric_no_viz
import copy


class TabularNaive(Naive):
    def _unpack_minibatch(self):
        # [ ( (cat, cont), y, [maybe task_id] ) ]
        x, y, *rest = self.mbatch

        # x is a tuple
        cat, cont = x

        # move to device
        cat  = cat.to(self.device, non_blocking=True)
        cont = cont.to(self.device, non_blocking=True)
        y    = y.to(self.device, non_blocking=True)

        if len(rest) > 0:
            t = rest[0].to(self.device, non_blocking=True)
            self.mbatch = ((cat, cont), y, t)
        else:
            self.mbatch = ((cat, cont), y)

    def forward(self):
        # mb_x is read-only and equals self.mbatch[0] -> (cat, cont)
        cat, cont = self.mb_x
        return self.model(cat, cont)
        
      
def load_dataset(config, dataset_name, classes, all_classes):

    data_path = f"dataset/{dataset_name}"

    train = np.load(f"{data_path}/train.npy")
    val   = np.load(f"{data_path}/val.npy")
    test  = np.load(f"{data_path}/test.npy")

    base_train = TabularDataset(train,
                                data_path+"/catfeaturelist.npy", 
                                classes, 
                                all_classes, 
                                fit=True)
    base_val   = TabularDataset(val, 
                                data_path+"/catfeaturelist.npy", 
                                classes, 
                                all_classes, 
                                fit=False,
                                scaler=base_train.scaler)
    base_test  = TabularDataset(test, 
                                data_path+"/catfeaturelist.npy", 
                                classes, 
                                all_classes, 
                                fit=False,
                                scaler=base_train.scaler)

    
    #train_dataset = AvalancheTabularDataset(base_train)
    #val_dataset = AvalancheTabularDataset(base_val)
    #test_dataset = AvalancheTabularDataset(base_test)
    
    return base_train, base_val, base_test

def get_dataset_info(config, dataset_name):
    """Reads dataset-specific information from the config."""
    try:
        dataset_config = config['datasets'][dataset_name]
        
        config['classes'] = tuple(dataset_config['classes'])
        config['num_class'] = dataset_config['num_class']
        config['model_path'] = f"model/Model_{dataset_name}.pth"
        
        return config
        
    except KeyError:
        print(f"Error: Configuration for dataset '{dataset_name}' not found in config.yaml.")
        exit()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['dataset_name'] = "CICIDS2017"
get_dataset_info(config, config['dataset_name'])

train_dataset, val_dataset, test_dataset = load_dataset(config, config['dataset_name'], config['classes'], config['classes'])

print(f"\nDataset A Info: {train_dataset.total_features} features, {config['num_class']} classes.")



# Create scenario to split dataset into task. Note: must be divisible by 10

scenario = nc_benchmark(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    n_experiences=4,       # split into 5 incremental tasks
    task_labels=False,     # class-incremental (no task id given)
    seed=0,
    fixed_class_order=list(range(config['num_class']))
)

balanced = lambda ds: split_validation_class_balanced(0.10, ds)  # 10% per class (approx)

scenario = benchmark_with_validation_stream(
    scenario,
    split_strategy=balanced      # class-balanced per experience
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TabTransformer(
    categories = train_dataset.vocab_sizes, 
    num_continuous = train_dataset.num_continuous_features,
    dim = 32,
    dim_out = config['num_class'],
    depth = 6, 
    heads = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.1, 
).to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

# Evaluation plugin to track metrics (accuracy + forgetting)

#def safe_log_single_metric(self, name, value, x_plot):
    # log as simple scalars
    #wandb.log({name: value, "x": x_plot})

WandBLogger.log_single_metric = log_single_metric_no_viz

interactive_logger = []
interactive_logger.append(InteractiveLogger())
interactive_logger.append(WandBLogger(
                            project_name="avalanche",
                            run_name=None, 
                            params={"mode": "disabled"}))
interactive_logger.append(TextLogger(open('log.txt', 'a')))


eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, trained_experience=True, stream=True),
    loss_metrics(epoch=True, stream=True),         
    forgetting_metrics(experience=True, stream=True),
    class_accuracy_metrics(experience=True),
    bwt_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=config['num_class'], save_image=False),
    loggers=interactive_logger                    
)



early_stop = EarlyStoppingPlugin(
    patience=1,
    val_stream_name='valid',
    metric_name='Loss_Stream',
    mode='min',
    peval_mode='epoch'
)


cl_strategy = TabularNaive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=config['batch_size'],
    train_epochs=config['epochs'],
    eval_mb_size=config['batch_size'],
    device=device,
    evaluator=eval_plugin                   # evaluate each epoch (feeds EarlyStopping)
)



def get_valid_epoch_loss(metrics):
    for k, v in metrics.items():
        if "Loss_Epoch" in k and "/valid_stream" in k:
            return float(v)
    raise KeyError("Validation epoch loss not found. Make sure loss_metrics(epoch=True, stream=False) is enabled.")


    
# Training loop over experiences
results = []
accuracies_after_each_exp = []   # store overall stream accuracy after each training experience

for experience in scenario.train_stream:
    print(f"\n--- Training on experience {experience.current_experience} -- classes: {experience.classes_in_this_experience}\n")
    cl_strategy.train(experience)

    print(" -> Validation on the validation stream (all experiences so far)...")
    current_val_exp = scenario.valid_stream[experience.current_experience]
    valid_results = cl_strategy.eval(current_val_exp)
    print(valid_results.items())
    val_loss = get_valid_epoch_loss(valid_results)
    
    print(" -> Testign on the test stream (all experiences so far)...")
    experience_eval_results = cl_strategy.eval(scenario.test_stream)
    # Avalanche prints detailed logs. We can collect key metrics from experience_eval_results if needed.
    # Instead, use the evaluation plugin outputs, or parse model accuracy manually:
    accuracies_after_each_exp.append(experience_eval_results)  # collect for analysis if desired'''




