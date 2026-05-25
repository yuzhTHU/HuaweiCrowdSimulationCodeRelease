<div align="center">

# Crowd Simulation with Diffusion Models

[![Documentation](https://img.shields.io/badge/Docs-Read%20Online-blue?style=for-the-badge&logo=read-the-docs&logoColor=white)](https://yuzhthu.github.io/HuaweiCrowdSimulationCode/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

</div>

This project is built on diffusion models and a Transformer architecture to simulate pedestrian trajectories in complex scenes. The codebase is well structured, thoroughly documented, supports training and evaluation on multiple mainstream datasets (ETH, UCY, SDD, etc.), and includes an interactive web-based visualization tool.

## 🛠 Installation

Set up the Python environment with the following steps:

```shell
# 1. Clone the repository
git clone git@github.com:yuzhTHU/HuaweiCrowdSimulationCode.git
cd HuaweiCrowdSimulationCode

# 2. Create and activate a Conda environment
conda create -p ./venv python=3.12 -y
conda activate ./venv

# 3. Install dependencies
# Core dependencies
pip install setproctitle tqdm numpy matplotlib pandas scipy torch ipykernel seaborn
# Web visualization dependencies
pip install fastapi "uvicorn[standard]" websockets
```

## 📂 Data Preparation

This project supports ETH, UCY, SDD, GC, WayMo, ORCA, and other datasets. The data directory is organized as follows:

  * `./data/ETH/`, `./data/UCY/`, ...: raw trajectory data.
  * `./data/.cache/`: preprocessed feature data used to accelerate loading.

### 1. Internal Users (FIB-Lab dl4 Access)

If you have access to our FIB-Lab `dl4` server, you can synchronize data and logs directly with the following `rsync` commands:

```shell
# Download data (including raw data and preprocessing cache)
mkdir -p ./data
rsync -anv --info=progress2 \
    --include='ETH/***' \
    --include='UCY/***' \
    --include='SDD/***' \
    --include='GC/***' \
    --include='WayMo/' \
    --include='WayMo/summary.csv' \
    --include='WayMo/Processed/***' \
    --include='ORCA/***' \
    --include='.cache/***' \
    --exclude='*' \
    dl4:~/WorkSpace/35-HuaweiCrowdSimulation/HuaweiCrowdSimulationCode/data/ ./data

# Download pretrained model logs (checkpoints)
mkdir -p ./logs
rsync -anv --info=progress2 \
    --exclude='***/.syncthing*' \
    dl4:~/WorkSpace/35-HuaweiCrowdSimulation/HuaweiCrowdSimulationCode/logs/ ./logs
```

### 2. External Users (Public Access)

If you do not have access to the internal server, you can download data through the dedicated `rsync` port below. Contact the administrator for the password.

```shell
rsync -av --port=8873 --exclude WayMo/Motion rsyncuser@dl4.yumeow.site::data_share ./data
# About 60 GB in total
# Remove --exclude WayMo/Motion to download the original WayMo tf data as well, about 87 GB
```

## 🚀 Training

### Start Training

Use `train.py` to launch training. The following example trains on all datasets and applies several data augmentation strategies (Drop Map/Goal/Speed):

```shell
python train.py \
    --name train_experiment \
    --datasets All \
    --test_ratio 0.2 \
    --p_drop_map 0.2 \
    --p_drop_destination 0.3 \
    --p_drop_speed 0.3
```

Logs, model weights (`checkpoint.pth`, `best.pth`), and training visualizations are saved under `logs/train/{yymmdd}_{name}_{hhmmss}_{hostname}/`.

### Resume Training

If training is interrupted unexpectedly, you can resume in either of the following ways:

1.  **Continue writing to the same directory** (recommended):
    ```shell
    python train.py --exp_name yymmdd_name_hhmmss_hostname
    ```
2.  **Load weights and write to a new directory**:
    ```shell
    python train.py \
        --name train_resume \
        --reload_checkpoint ./logs/train/yymmdd_name_hhmmss_hostname/checkpoint.pth
    ```

## ⚡ Evaluation

Use `test.py` for inference and evaluation. This script loads a trained model, generates future trajectories, and computes metrics such as ADE and FDE.

```shell
python test.py \
    --reload_checkpoint /path/to/logs/train/xxx_train_xxx/best.pth \
    --datasets All \
    --test_ratio 0.2
```

  * `--roll_step`: number of consecutive prediction steps (simulation horizon).
  * `--sample_num`: number of samples drawn for each trajectory (used to evaluate diversity).

Evaluation results and visualization images are saved in `logs/test/xxx_sample_xxx`.

## 🎨 Visualization

This project provides an interactive web-based visualization platform for real-time simulation inspection.

```shell
uvicorn app:app --host 0.0.0.0 --port 12345
```

After startup, open `http://localhost:12345` in your browser (or the corresponding server IP).
**Features include**:

  * Load model checkpoints and their corresponding configs.
  * Load and switch datasets.
  * Visualize real trajectories and maps.
  * Run real-time simulation and adjust trail length and simulation duration.

## 📖 Documentation

The project includes detailed design documentation and API references.

1.  **View locally**:
    Open `./docs/build/html/index.html` in a browser.

2.  **View remotely**:
    In a remote server environment, start a temporary HTTP server:

    ```shell
    python -m http.server 8000 -d ./docs/build/html
    ```

    Then open `http://localhost:8000` in your local browser.

## 📊 Code Statistics

This project follows high-quality engineering standards. The core code has roughly a 30% comment ratio, with clearly defined input and output interfaces.

```text
❯ pygount --format=summary --folders-to-skip=__pycache__,web ./src
┏━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ Language      ┃ Files ┃     % ┃ Code ┃    % ┃ Comment ┃    % ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ Python        │    32 │  91.4 │ 2742 │ 58.2 │    1364 │ 29.0 │
│ __empty__     │     1 │   2.9 │    0 │  0.0 │       0 │  0.0 │
│ __duplicate__ │     2 │   5.7 │    0 │  0.0 │       0 │  0.0 │
├───────────────┼───────┼───────┼──────┼──────┼─────────┼──────┤
│ Sum           │    35 │ 100.0 │ 2742 │ 58.2 │    1364 │ 29.0 │
└───────────────┴───────┴───────┴──────┴──────┴─────────┴──────┘
```

## System and Hardware

### Linux -> Windows

This codebase was developed on Ubuntu 22.04, but it has also been tested to run on Windows 10 without modification.

### NVIDIA GPU -> Ascend NPU

This codebase was originally developed on NVIDIA GPUs. To run it on a HUAWEI Ascend NPU, set the environment variable with `export USE_NPU=True`; the code will then switch to the NPU automatically.

Note that the NPU does not yet support the `_native_multi_head_attention` operator. As a result, the `MultiheadAttention` module falls back to the unoptimized MatMul path during inference. This does not affect training speed (`model.train()` mode), but it does reduce inference speed (`model.eval()` mode) by roughly 2x to 3x.
