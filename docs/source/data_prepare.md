# 数据准备指南 (Data Preparation Guide)

本文档旨在指导用户如何准备用于 **Crowd Simulation Model** 的输入数据。为了满足不同用户的需求，本模型设计了两种灵活的数据输入方式：

1.  **原始数据模式 (Raw Data Mode)**：如果您拥有标准的轨迹表格（CSV/Pandas DataFrame），可以直接使用内置的流水线进行处理。这是最简单的上手方式。
2.  **高级自定义模式 (Advanced/Direct Mode)**：如果您希望深度定制输入特征（例如手动指定速度、目的地或历史轨迹），您可以直接构建模型所需的张量（Tensors）。

-----

## 1\. 原始数据模式 (Raw Data Mode)

在这种模式下，您只需准备**轨迹列表**和**地图信息**。模型配套的预处理代码（参考 `sample.py`）会自动完成速度计算、历史回溯和张量构建。

### 1.1 轨迹数据格式

请准备一个 `pandas.DataFrame`，其中每一行代表一个智能体在某一帧的状态。必须包含以下列：

| 列名 (Column) | 类型 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| **`f`** | int | 帧索引 (Frame Index)，需为单调递增的整数 | `0`, `1`, `2`... |
| **`id`** | int/str | 智能体唯一标识符 | `1`, `102`, `"ped_0"` |
| **`type`** | str | 智能体类型，必须为 `'pedestrian'` 或 `'vehicle'` | `"pedestrian"` |
| **`x`** | float | 世界坐标系下的 X 坐标 (米) | `12.5` |
| **`y`** | float | 世界坐标系下的 Y 坐标 (米) | `-3.4` |

**提示**：

  * 数据的采样频率应与模型配置参数 `args.fps` 一致（默认 2.5Hz），否则请先进行重采样。
  * 坐标系方向需与地图数据一致。

### 1.2 地图数据格式

地图需要被封装为 `RasterizedMap` 对象，包含栅格数据（numpy array）和物理边界信息。

```python
from src.dataset.base_dataset import RasterizedMap
import numpy as np

# 1. 准备栅格地图 (0: 可通行区域/空地, 1: 障碍物)
# 形状为 (W, H) 的二维数组
grid_map = np.zeros((100, 100)) 

# 2. 定义该地图对应的物理世界边界 (单位: 米)
map_data = RasterizedMap(
    map=grid_map,
    xmin=0.0,  # 地图最左侧对应的 x 坐标
    xmax=50.0, # 地图最右侧对应的 x 坐标
    ymin=0.0,  # 地图最下方对应的 y 坐标
    ymax=50.0  # 地图最上方对应的 y 坐标
)
```

-----

## 2\. 数据处理流水线 (Processing Pipeline)

如果您提供的是上述 **原始数据**，系统会通过一个链式处理过程将其转换为模型输入。了解这一过程有助于您理解模型实际上利用了哪些信息，或者为您切换到“高级模式”做准备。

以下逻辑基于 `sample.py` 中的实现：

1.  **筛选 (Filtering)**：
    根据当前模拟的起始帧 `frame_idx`，系统会筛选出当前时刻存在的行人 (`ped_list`) 和车辆 (`veh_list`)。

2.  **当前位置 (`pos`)**：
    直接提取 `frame_idx` 时刻所有行人的 `(x, y)` 坐标。

      * 结果形状：`(#ped, 2)`

3.  **当前速度 (`vel`)**：
    通过有限差分计算：`(Pos_t - Pos_{t-1}) * FPS`。

      * 结果形状：`(#ped, 2)`

4.  **历史轨迹 (`hst`)**：
    回溯提取过去 `args.hist_step` 帧（例如过去 8 帧）的位置数据。

      * **注意**：在 `sample.py` 的逻辑中，`hst` 通常**不包含**当前帧 `frame_idx`，而是截止到 `frame_idx - 1`。
      * 结果形状：`(#ped, hist_step, 2)`

5.  **车辆轨迹 (`veh`)**：
    提取车辆的历史轨迹。与行人不同，车辆历史通常**包含**当前帧 `frame_idx`。

      * 结果形状：`(#veh, hist_step + 1, 2)`

6.  **目的地推断 (`des`)**：
    系统默认选取该 ID 在整个 DataFrame 中出现的**最后时刻**的位置作为其潜在目的地。

      * 结果形状：`(#ped, 2)`

7.  **期望速率 (`spd`)**：
    计算未来一段时间（如 5 秒）内的平均移动速率标量。

      * 结果形状：`(#ped, 1)`

-----

## 3\. 高级自定义模式 (Advanced/Direct Mode)

如果您已经有预处理好的数据，或者希望测试一些假设（例如：“如果目的地在别处，模型会怎么走？”），您可以跳过 DataFrame 构建环节，直接构造 Tensor 输入模型。

请准备以下 `torch.FloatTensor` 格式的变量，并确保它们在 GPU/CPU 上与模型一致。

**维度说明**：

  * `B`: Batch Size（通常为 1 或 `sample_num`）
  * `N`: 当前场景中的行人数
  * `M`: 当前场景中的车辆数
  * `H`: 历史步长 (`args.hist_step`)

| 变量名 | 形状 (Shape) | 物理含义与约束 | 自定义建议 |
| :--- | :--- | :--- | :--- |
| **`pos`** | `(B, N, 2)` | **当前位置** $(x, y)$。 | 必须准确对应地图坐标系。 |
| **`vel`** | `(B, N, 2)` | **当前速度** $(v_x, v_y)$，单位 m/s。 | 既然是自定义，您可以尝试修改此值来观察模型对初始冲量的反应。 |
| **`hst`** | `(B, N, H, 2)` | **行人历史轨迹**。通常不包含当前帧。 | 如果数据缺失，可以用当前位置填充或线性插值。 |
| **`des`** | `(B, N, 2)` | **目的地坐标**。 | **这是最常用的控制变量**。修改此变量可引导模型生成前往特定区域的轨迹。 |
| **`spd`** | `(B, N, 1)` | **期望速率** (标量)。 | 控制代理移动的急切程度。 |
| **`veh`** | `(B, M, H+1, 2)` | **车辆轨迹**。包含当前帧。 | 如果场景中无车，该 Tensor 可以是空的或特定填充处理。 |
| **`map`** | `(W, H)` | **地图特征图**。 | 0 为空地，1 为障碍物。模型会根据 `xmin/xmax` 等参数将其映射到物理空间。 |

### 代码调用示例

```python
import torch
from src.model.model import Model

# 1. 初始化模型
model = Model(args)
# 加载权重...

# 2. 准备数据 (Tensor)
# 假设您已经手动构建了符合上述形状的 Tensor
pos_tensor = ... 
vel_tensor = ...
# ...

# 3. 注入数据到模型
# 注意：必须先设置 Map 和 Vehicle，再设置 Pedestrian
model.set_map_embedding(
    map=map_tensor, 
    xmin=0.0, xmax=100.0, 
    ymin=0.0, ymax=100.0
)
model.set_veh_embedding(veh=veh_tensor)

model.set_ped_embedding(
    pos=pos_tensor, 
    vel=vel_tensor, 
    hst=hst_tensor, 
    des=des_tensor, 
    spd=spd_tensor
)

# 4. 计算局部环境特征 (必须在 set_map 和 set_ped 之后调用)
model.set_sur_info() 

# 5. 开始推理 (例如在扩散循环中)
output = model(
    noisy_acc=noisy_input, 
    denoise_t=t, 
    ped_length=ped_len, 
    veh_length=veh_len
)
```

通过这种方式，您可以完全绕过数据加载器，灵活控制每一个输入变量，实现高度定制化的模拟实验。