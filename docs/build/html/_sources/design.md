# 华为人群模拟项目 - 设计与架构文档

## 1\. 项目概述 (Overview)

本项目旨在基于扩散模型（Diffusion Model）实现高保真的多智能体行人轨迹模拟。系统能够接受场景地图、行人历史轨迹作为输入，预测并在网页端可视化未来的人群移动趋势。项目包含模型训练、推理采样以及基于 Web 的交互式可视化三大模块。

## 2\. 系统架构 (System Architecture)

### 2.1 核心模块划分

  * **Dataset (`src.dataset`)**: 负责多源异构数据的标准化加载。支持 ETH, UCY, SDD, GC, WayMo, ORCA 等主流数据集。核心类 `BaseDataset` 实现了统一的滑动窗口采样、坐标归一化和数据缓存机制。
  * **Model (`src.model`)**:
      * `Model`: 基础 Transformer 模型，包含对行人、车辆、地图的 Attention 机制。
      * `RelativeModel`: 采用相对坐标编码的改进模型，具有更好的泛化性。
      * `NewModel`: 最新的实验性模型架构。
  * **Diffusion (`src.diffusion`)**: 实现了 DDPM (概率扩散) 和 DDIM (隐式扩散) 两种采样策略，负责将高斯噪声逐步还原为符合物理规律的轨迹加速度。
  * **Web (`src.web`)**: 基于 FastAPI 和 WebSockets 的可视化后端，支持实时推流模拟结果到前端。

### 2.2 数据流向 (Data Flow)

1.  **输入**: 原始轨迹 CSV/Txt 文件 + 场景图片。
2.  **预处理**: `BaseDataset` 进行坐标变换 (Homography)、重采样 (Resample) 和归一化。
3.  **训练**: `train.py` 使用滑动窗口数据训练扩散模型，预测未来 `pred_step` 帧的加速度。
4.  **推理**: `sample.py` 或 `simulate.py` 利用训练好的模型，结合社会力引导 (Social Force Guidance) 进行多帧自回归预测 (Rollout)。
5.  **展示**: 结果通过 WebSocket 发送至前端，使用 Plotly.js 在地图上绘制轨迹。

## 3\. 关键算法设计

### 3.1 扩散过程

采用 **DDIM** 进行快速采样。模型预测的是 **加速度 (Acceleration)** 而非直接的位置，这保证了轨迹的平滑性和物理合理性。

### 3.2 引导策略 (Guidance)

为了增强模拟的可控性，在去噪过程中引入了 Classifier-Free Guidance (CFG) 和基于梯度的能量引导：

  * **目的地引导**: 引导行人向预设终点移动。
  * **避障引导**: 利用 `get_force_map` 计算地图势能场，排斥行人远离障碍物。

## 4\. 接口说明 (API Interface)

### 4.1 训练接口

```bash
python train.py --name [实验名] --datasets [数据集列表] --loss_type [noise|accelerate]
```

### 4.2 Web 模拟接口

Web 后端监听 `ws://0.0.0.0:12345/ws`。

  * **Start Action**: `{ "action": "start", "dataset_name": "ETH", "frame_idx": 100, "frame_num": 200 }`
  * **Response**: 包含每一帧的行人 ID、类型和 (x, y) 坐标流。
