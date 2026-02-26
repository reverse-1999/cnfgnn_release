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

### 我的批注 ：pip list 显示的包
Package                   Version
------------------------- ----------------
absl-py                   2.4.0
aiohappyeyeballs          2.6.1
aiohttp                   3.13.3
aiosignal                 1.4.0
annotated-types           0.7.0
anyio                     4.12.1
argon2-cffi               25.1.0
argon2-cffi-bindings      25.1.0
arrow                     1.4.0
asttokens                 3.0.1
async-lru                 2.2.0
async-timeout             5.0.1
attrs                     25.4.0
babel                     2.18.0
beautifulsoup4            4.14.3
bleach                    6.3.0
cachetools                4.2.4
certifi                   2026.1.4
cffi                      2.0.0
charset-normalizer        3.4.4
click                     8.3.1
colorama                  0.4.6
comm                      0.2.3
contourpy                 1.3.2
cycler                    0.12.1
debugpy                   1.8.20
decorator                 5.2.1
defusedxml                0.7.1
exceptiongroup            1.3.1
executing                 2.2.1
fastjsonschema            2.21.2
filelock                  3.24.3
fonttools                 4.61.1
fqdn                      1.5.1
frozenlist                1.8.0
fsspec                    2026.2.0
future                    1.0.0
gitdb                     4.0.12
gitpython                 3.1.46
google-auth               1.35.0
google-auth-oauthlib      0.4.6
grpcio                    1.78.1
h11                       0.16.0
httpcore                  1.0.9
httpx                     0.28.1
idna                      3.11
ipykernel                 7.2.0
ipython                   8.38.0
ipywidgets                8.1.8
isoduration               20.11.0
jedi                      0.19.2
jinja2                    3.1.6
joblib                    1.5.3
json5                     0.13.0
jsonpointer               3.0.0
jsonschema                4.26.0
jsonschema-specifications 2025.9.1
jupyter                   1.1.1
jupyter-client            8.8.0
jupyter-console           6.6.3
jupyter-core              5.9.1
jupyter-events            0.12.0
jupyter-lsp               2.3.0
jupyter-server            2.17.0
jupyter-server-terminals  0.5.4
jupyterlab                4.5.4
jupyterlab-pygments       0.3.0
jupyterlab-server         2.28.0
jupyterlab-widgets        3.0.16
kiwisolver                1.4.9
lark                      1.3.1
lightning-utilities       0.15.3
markdown                  3.10.2
markupsafe                3.0.3
matplotlib                3.10.8
matplotlib-inline         0.2.1
mistune                   3.2.0
mpmath                    1.3.0
multidict                 6.7.1
nbclient                  0.10.4
nbconvert                 7.17.0
nbformat                  5.10.4
nest-asyncio              1.6.0
networkx                  3.4.2
notebook                  7.5.3
notebook-shim             0.2.4
numpy                     1.23.5
oauthlib                  3.3.1
overrides                 7.7.0
packaging                 26.0
pandas                    2.3.3
pandocfilters             1.5.1
parso                     0.8.6
pillow                    12.1.1
platformdirs              4.9.2
prometheus-client         0.24.1
prompt-toolkit            3.0.52
propcache                 0.4.1
protobuf                  3.20.1
psutil                    7.2.2
pure-eval                 0.2.3
pyasn1                    0.6.2
pyasn1-modules            0.4.2
pycparser                 3.0
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
referencing               0.37.0
requests                  2.32.5
requests-oauthlib         2.0.0
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rfc3987-syntax            1.1.0
rpds-py                   0.30.0
rsa                       4.9.1
scikit-learn              1.7.2
scipy                     1.15.3
send2trash                2.1.0
sentry-sdk                2.53.0
setuptools                59.5.0
six                       1.17.0
smmap                     5.0.2
soupsieve                 2.8.3
stack-data                0.6.3
sympy                     1.13.1
tensorboard               2.2.0
tensorboard-data-server   0.7.2
tensorboard-plugin-wit    1.8.1
tensorboardx              2.6.4
terminado                 0.18.1
threadpoolctl             3.6.0
tinycss2                  1.4.0
tomli                     2.4.0
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
tqdm                      4.67.3
traitlets                 5.14.3
typing-extensions         4.15.0
typing-inspection         0.4.2
tzdata                    2025.3
uri-template              1.3.0
urllib3                   2.6.3
wandb                     0.25.0
wcwidth                   0.6.0
webcolors                 25.10.0
webencodings              0.5.1
websocket-client          1.9.0
werkzeug                  3.1.6
wheel                     0.46.3
widgetsnbextension        4.0.15
xxhash                    3.6.0
yarl                      1.22.0