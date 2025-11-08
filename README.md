# Repository for Continual Learning Benchmarks (EWC, iCaRL, Oracle)

An educational repository for running small Continual Learning (CL) benchmarks on tabular network intrusion data. It focuses on three methods:

- Oracle: a standard PyTorch training loop (`oracle.py`)
- EWC: Elastic Weight Consolidation (`ewc.ipynb`)
- iCaRL: Incremental Classifier and Representation Learning (`icarl.ipynb`)

Under the hood we use [Avalanche](https://avalanche.continualai.org) for CL strategies and metrics.

For a more interactive documentation, please refer to deepwiki
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/afifhaziq/IDSContinualBenchmark)

### Quick start

1) Install uv (recommended)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternatively with pipx
pip install uv
```

2) Clone the repository

```bash
git clone https://github.com/afifhaziq/IDSContinualBenchmark.git
cd IDSContinualBenchmark
```

3) Sync the environment with uv

This project expects a `pyproject.toml`. If it’s present, simply run:

```bash
uv sync
```

If you don’t see a `pyproject.toml`, create one and add the dependencies, then sync:

```bash
# Create a minimal pyproject, then add deps
uv init

# CPU wheels (simple):
uv add avalanche-lib torch torchvision torchaudio numpy scikit-learn tqdm wandb jupyter ipywidgets torchsummary

# OR for CUDA wheels (replace cu121 with your CUDA version):
uv add --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
uv add avalanche-lib numpy scikit-learn tqdm wandb jupyter ipywidgets torchsummary

# Install exactly as pinned in uv.lock / pyproject
uv sync
```

uv will create and manage a local virtual environment at `.venv/`. You can use it either by activating it or by prefixing commands with `uv run`.

```bash
# Optional: activate the venv
source .venv/bin/activate

# uv enable us to run the code directly without activating the venv such as this example
uv run oracle.py
```


#### Enable Jupyter to detect the virtual environment

```bash
python -m ipykernel install --user --name=IDSContinualBenchmark --display-name "IDSContinualBenchmark"
```
Note: use this command If youre Jupyter did not detect the venv from the kernel selection.

### Dataset setup

All examples assume the CICIDS2017 data has been preprocessed into three NumPy arrays, placed as:

```
CICIDS2017/
  train.npy
  test.npy
  val.npy   # used by Oracle and EWC notebook
```

Expected format:

- Each `.npy` is a 2D array shaped `[num_samples, num_features + 1]`.
- The last column is the integer class label in `[0..7]` (CICIDS2017).
- Feature columns should be standardized (the notebooks/scripts also demonstrate doing this with `StandardScaler`). This is optional. you can skip.

Default class names (see `oracle.py`):

1. Benign
2. DoS GoldenEye
3. DoS Hulk
4. DoS Slowhttptest
5. DoS slowloris
6. FTP-Patator
7. Heartbleed
8. SSH-Patator

Note: Change accordingly based on dataset class names

### Baselines and how to run

#### 1) Oracle (standard PyTorch training)

`oracle.py` provides a conventional training/validation loop with early stop and model checkpointing.

```bash
# From the repo root
uv run python oracle.py
```

Outputs:
- Saves best model weights to `Oracle_LiteNet.pth`.
- Prints final accuracy, classification report, and confusion matrix on the test set.

You can adjust hyperparameters (epochs, LR, batch size) and class names directly in `oracle.py`.


#### 2) EWC (Elastic Weight Consolidation)

Open and execute `ewc.ipynb` in Jupyter. It creates a class-incremental benchmark with Avalanche and trains using EWC.

```bash
uv run jupyter lab
# then open ewc.ipynb
```

Notes:
- The notebook expects `CICIDS2017/train.npy`, `test.npy`, and `val.npy` to exist.
- Tune `n_experiences`, `fixed_class_order`, batch sizes, epochs, and `ewc_lambda` as needed.
- The notebook includes optional Weights & Biases logging; comment those lines if you don’t want to log externally.


#### 3) iCaRL (Incremental Classifier and Representation Learning)

Open and execute `icarl.ipynb` in Jupyter. It sets up a feature extractor + classifier and trains with the iCaRL strategy in Avalanche.

```bash
uv run jupyter lab
# then open icarl.ipynb
```

Notes:
- The notebook expects `CICIDS2017/train.npy` and `test.npy` (validation is not used in the minimal example).
- Tune memory size, epochs, batch sizes, and optimizer settings. Please explore this one.
- Optional W&B logging can be disabled by commenting the corresponding lines.


### Project structure (key files)

- `oracle.py`: Standard PyTorch baseline training + evaluation.
- `ewc.ipynb`: EWC continual learning experiment with Avalanche.
- `icarl.ipynb`: iCaRL continual learning experiment with Avalanche.
- `preprocess.py`: Helpers to build PyTorch `DataLoader`s from NumPy arrays.
- `icarlmodel.py`: Network definitions used by the iCaRL notebook.


### Tips and troubleshooting

- uv docs. Refer here if you are unsure how to use it.
  Refer to the uv docs: [Getting Started — Features](https://docs.astral.sh/uv/getting-started/features).

- Jupyter tqdm bar shows “IProgress not found”:
  ```bash
  uv add ipywidgets
  ```

- W&B reports missing `nbformat` when logging from notebooks:
  ```bash
  uv add nbformat
  ```

- CUDA wheels for PyTorch: prefer installing from the official CUDA index URL (replace `cu121` with your CUDA version):
  ```bash
  uv add --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```

- GPU not used: ensure a CUDA build is installed and `torch.cuda.is_available()` returns `True`.


### Acknowledgements

- [Avalanche](https://avalanche.continualai.org) — continual learning library used here.


### License

MIT License

Copyright (c) 2025 afifhaziq

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


