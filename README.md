# Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling

This repository is the official PyTorch implementation of "Cross-Node Federated Graph Neural Network for Spatio-Temporal Modeling".

## Setup

### Environment

```bash
conda create -n fedgnn "python<3.8"
conda activate fedgnn
bash install.sh
```

### Data

Download `data.tar.bz2`. Then extract it to the root directory of the repository:

```bash
tar -xjf data.tar.bz2
```

## Experiments

### Main Experiments

`submission_exps/exp_main.sh` contains all commands used for experiments in Table 2 and Table 3.

### Inductive Learning on Unseen Nodes

Run `python submission_exps/exp_inductive.py` to print all commands for Table 4.

### Ablation Study: Effect of Alternating Training of Node-Level and Spatial Models

Run `python submission_exps/exp_at.py` to print all commands for Figure 2.

### Ablation Study: Effect of Client Rounds and Server Rounds

Run `python submission_exps/exp_crsr.py` to print all commands for Figure 3.

### 我的批注 ：pip list 
Package                   Version
------------------------- ----------------
markdown                  3.10.2
markupsafe                3.0.3
matplotlib                3.10.8
matplotlib-inline         0.2.1
numpy                     1.23.5
pandas                    2.3.3
pydantic                  2.12.5
pydantic-core             2.41.5
pydeprecate               0.3.1
pygments                  2.19.2
pyparsing                 3.3.2
python-dateutil           2.9.0.post0
python-json-logger        4.0.0
pytorch-lightning         1.5.10
pytz                      2025.2
pywinpty                  3.0.3
pyyaml                    6.0.3
pyzmq                     27.1.0
scikit-learn              1.7.2
scipy                     1.15.3
torch                     2.6.0+cu126
torch-cluster             1.6.3+pt26cu126
torch-geometric           2.7.0
torch-scatter             2.1.2+pt26cu126
torch-sparse              0.6.18+pt26cu126
torch-spline-conv         1.2.2+pt26cu126
torchaudio                2.6.0
torchmetrics              1.8.2
torchvision               0.21.0+cu126
tornado                   6.5.4
wandb                     0.25.0