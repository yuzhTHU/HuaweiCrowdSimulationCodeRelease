<div align="center">

# Crowd Simulation with Diffusion Models

[![Documentation](https://img.shields.io/badge/Docs-Read%20Online-blue?style=for-the-badge&logo=read-the-docs&logoColor=white)](https://yuzhthu.github.io/HuaweiCrowdSimulationCode/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**[English](README_en.md)** | **[简体中文](README.md)**

</div>

This project is based on Diffusion Models and Transformer architecture, aiming to simulate pedestrian movement trajectories in complex scenarios. The codebase is well-structured with comprehensive comments, supports training and testing on multiple mainstream datasets (ETH, UCY, SDD, etc.), and provides a web-based interactive visualization tool.

## 🛠 Installation

Please follow the steps below to configure the Python environment:

```shell
# 1. Clone the repository
git clone git@github.com:yuzhTHU/HuaweiCrowdSimulationCode.git
cd HuaweiCrowdSimulationCode

# 2. Create and activate Conda environment
conda create -p ./venv python=3.12 -y
conda activate ./venv

# 3. Install dependencies
# Core code dependencies
pip install setproctitle tqdm numpy matplotlib pandas scipy torch ipykernel seaborn
# Web visualization dependencies
pip install fastapi "uvicorn[standard]" websockets
```

## 📂 Data Preparation

This project supports multiple datasets including ETH, UCY, SDD, GC, WayMo, and ORCA. The data directory structure is as follows:

  * `./data/ETH/`, `./data/UCY/`, ... : Raw trajectory data.
  * `./data/.cache/` : Preprocessed feature data (for faster loading).

### 1. Internal Users (FIB-Lab dl4 Access)

If you have access to the FIB-Lab dl4 server, you can directly run the following `rsync` commands to sync data and logs:

```shell
# Download data (including raw data and preprocessed cache)
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

# Download pretrained model checkpoints
mkdir -p ./logs
rsync -anv --info=progress2 \
    --exclude='***/.syncthing*' \
    dl4:~/WorkSpace/35-HuaweiCrowdSimulation/HuaweiCrowdSimulationCode/logs/ ./logs
```

### 2. External Users (Public Access)

If you don't have access to the internal server, you can download data via the following dedicated rsync port (contact administrator for password):

```shell
rsync -av --port=8873 --exclude WayMo/Motion rsyncuser@dl4.yumeow.site::data_share ./data
# Approximately 60 GB total
# Remove --exclude WayMo/Motion to get WayMo raw tf data, approximately 87GB
```

## 🚀 Training

### Start Training

Use `train.py` to start training. The following example demonstrates training on all datasets with data augmentation strategies (Drop Map/Goal/Speed):

```shell
python train.py \
    --name train_experiment \
    --datasets All \
    --test_ratio 0.2 \
    --p_drop_map 0.2 \
    --p_drop_destination 0.3 \
    --p_drop_speed 0.3
```

Running logs, model parameters (`checkpoint.pth`, `best.pth`), and visualization results will be saved in `logs/train/{yymmdd}_{name}_{hhmmss}_{hostname}/`.

### Resume Training

If training is accidentally interrupted, you can resume in two ways:

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

Use `test.py` for inference and evaluation. This script loads the trained model, generates future trajectories, and calculates metrics such as ADE/FDE.

```shell
python test.py \
    --reload_checkpoint /path/to/logs/train/xxx_train_xxx/best.pth \
    --datasets All \
    --test_ratio 0.2
```

  * `--roll_step`: Number of steps for continuous prediction (simulation duration).
  * `--sample_num`: Number of samples per trajectory (for evaluating generation diversity).

Test results and visualization images will be saved in `logs/test/xxx_sample_xxx`.

## 🎨 Visualization

This project provides a web-based interactive visualization platform for real-time simulation viewing.

```shell
uvicorn app:app --host 0.0.0.0 --port 12345
```

After starting, access `http://localhost:12345` (or the server's corresponding IP) in your browser.

**Features include**:

  * Load model weights and corresponding configurations.
  * Load and switch between datasets.
  * Visualize trajectories and maps from real data.
  * Run real-time simulations, adjust trail length and simulation duration.

## 📖 Documentation

The project includes detailed design documents and API interface descriptions.

1.  **View locally**:
    Open `./docs/build/html/index.html` in your browser.

2.  **View remotely**:
    If in a remote server environment, start a temporary HTTP service:

    ```shell
    python -m http.server 8000 -d ./docs/build/html
    ```

    Then access `http://localhost:8000` in your local browser.

## 📊 Code Statistics

This project follows high-quality engineering standards, with approximately 30% of core code being comments, clearly defining input/output interfaces.

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

This codebase is developed on Ubuntu 22.04, but has been tested to run directly on Windows 10 without any modifications.

### NVIDIA GPU -> Ascend NPU

This codebase is developed for NVIDIA GPUs. To run on HUAWEI Ascend NPU, set the environment variable via `export USE_NPU=True`, and the code will automatically switch to run on NPU.

Note that since NPU does not support the `_native_multi_head_attention` operator yet, the `MultiheadAttention` module can only fallback to the unoptimized `MatMul` operator during inference. This does not affect training speed (i.e., in `model.train()` state), but will significantly reduce inference speed (i.e., in `model.eval()` state) by 2~3 times.
