# Crowd Simulation with Diffusion Models

本项目基于扩散模型（Diffusion Models）与 Transformer 架构，旨在模拟复杂场景下的行人移动轨迹。项目代码结构清晰，注释详尽，支持多种主流数据集（ETH, UCY, SDD, etc.）的训练与测试，并提供了基于 Web 的交互式可视化工具。

## 🛠 环境安装 (Installation)

请按照以下步骤配置 Python 环境：

```shell
# 1. 克隆代码库
git clone git@github.com:yuzhTHU/HuaweiCrowdSimulationCode.git
cd HuaweiCrowdSimulationCode

# 2. 创建并激活 Conda 环境
conda create -p ./venv python=3.12 -y
conda activate ./venv

# 3. 安装依赖
# 核心代码依赖
pip install setproctitle tqdm numpy matplotlib pandas scipy torch ipykernel seaborn
# web 可视化依赖
pip install fastapi "uvicorn[standard]" websockets
```

## 📂 数据准备 (Data Preparation)

本项目支持 ETH, UCY, SDD, GC, WayMo, ORCA 等多种数据集。数据目录结构如下：

  * `./data/ETH/`, `./data/UCY/`, ... : 原始轨迹数据。
  * `./data/.cache/` : 经过预处理后的特征数据（用于加速加载）。

### 1\. 内部用户 (FIB-Lab dl4 Access)

如果您拥有 FIB-Lab dl4 服务器的访问权限，可以直接运行下述 `rsync` 命令同步数据和日志：

```shell
# 下载数据 (包含原始数据和预处理缓存)
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

# 下载预训练模型日志 (Checkpoints)
mkdir -p ./logs
rsync -anv --info=progress2 \
    --exclude='***/.syncthing*' \
    dl4:~/WorkSpace/35-HuaweiCrowdSimulation/HuaweiCrowdSimulationCode/logs/ ./logs
```

### 2\. 外部用户 (Public Access)

如果您没有内部服务器权限，可以通过以下专用 rsync 端口下载数据（请联系管理员获取密码）：

```shell
rsync -av --port=8873 rsyncuser@dl4.yumeow.site::data_share ./data
```

## 🚀 模型训练 (Training)

### 开始训练

使用 `train.py` 启动训练。以下示例展示了在所有数据集上进行训练，并应用了一定的数据增强策略（Drop Map/Goal/Speed）：

```shell
python train.py \
    --name train_experiment \
    --datasets All \
    --test_ratio 0.2 \
    --p_drop_map 0.2 \
    --p_drop_destination 0.3 \
    --p_drop_speed 0.3
```

运行日志、模型参数（`checkpoint.pth`, `best.pth`）和训练过程的可视化结果将被保存在 `logs/train/{yymmdd}_{name}_{hhmmss}_{hostname}/` 目录中。

### 恢复训练 (Resume)

如果训练意外中断，可以通过以下两种方式恢复：

1.  **继续写入同一目录**（推荐）：
    ```shell
    python train.py --exp_name yymmdd_name_hhmmss_hostname
    ```
2.  **加载权重并写入新目录**：
    ```shell
    python train.py \
        --name train_resume \
        --reload_checkpoint ./logs/train/yymmdd_name_hhmmss_hostname/checkpoint.pth
    ```

## ⚡ 模型测试 (Evaluation)

使用 `test.py` 进行推理和评估。该脚本会加载训练好的模型，生成未来的轨迹并计算 ADE/FDE 等指标。

```shell
python test.py \
    --name sample_test \
    --reload_checkpoint /path/to/logs/train/xxx_train_xxx/best.pth \
    --roll_step 500 \
    --sample_num 10
```

  * `--roll_step`: 连续预测的步数（模拟时长）。
  * `--sample_num`: 对每条轨迹采样的次数（用于评估生成的多样性）。

测试结果和可视化图片将保存在 `logs/test/xxx_sample_xxx` 中。

## 🎨 可视化 (Visualization)

本项目提供了一个基于 Web 的交互式可视化平台，用于实时查看模拟效果。

```shell
uvicorn app:app --host 0.0.0.0 --port 12345
```

启动后，请在浏览器中访问 `http://localhost:12345`（或服务器对应的 IP）。
**功能包括**：

  * 加载模型权重与对应配置。
  * 加载并切换数据集。
  * 可视化真实数据的轨迹与地图。
  * 运行实时模拟，调整拖尾长度和模拟时长。

## 📖 文档 (Documentation)

项目包含详细的设计文档与 API 接口说明。

1.  **直接查看**：
    请在浏览器中打开本地文件 `./docs/build/html/index.html`。

2.  **远程查看**：
    如果是远程服务器环境，可启动一个临时的 HTTP 服务：

    ```shell
    python -m http.server 8000 -d ./docs/build/html
    ```

    然后在本地浏览器访问 `http://localhost:8000`。

## 📊 代码统计 (Code Statistics)

本项目遵循高质量工程标准，核心代码注释比例约为 30%，清晰定义了输入输出接口。

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