import numpy as np
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

def preprocess(train, test, val, batch_size):

    scaler = StandardScaler().fit(train[:, :-1])

    train_x = scaler.transform(train[:, :-1])
    test_x  = scaler.transform(test[:, :-1])
    train_y = train[:, -1].astype(np.int64)
    test_y  = test[:, -1].astype(np.int64)
    val_x   = scaler.transform(val[:, :-1])
    val_y   = val[:, -1].astype(np.int64)

    
    train_dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y)) 
    test_dataset = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y))
    val_dataset = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y))
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, test_loader, val_loader