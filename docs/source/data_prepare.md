# Data Preparation Guide

This document explains how to prepare input data for the **Crowd Simulation Model**. To support different use cases, the model provides two flexible data input modes:

1.  **Raw Data Mode**: If you already have standard trajectory tables (CSV/Pandas DataFrame), you can process them directly with the built-in pipeline. This is the easiest way to get started.
2.  **Advanced/Direct Mode**: If you want deep control over input features, such as manually specifying velocity, destination, or trajectory history, you can construct the tensors required by the model directly.

-----

## 1\. Raw Data Mode

In this mode, you only need to prepare **trajectory records** and **map information**. The preprocessing pipeline provided by the project (see `sample.py`) will automatically compute velocities, trace back history, and build tensors.

### 1.1 Trajectory Data Format

Prepare a `pandas.DataFrame` in which each row represents the state of one agent in one frame. It must contain the following columns:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`f`** | int | Frame index; must be a monotonically increasing integer | `0`, `1`, `2`... |
| **`id`** | int/str | Unique agent identifier | `1`, `102`, `"ped_0"` |
| **`type`** | str | Agent type; must be either `'pedestrian'` or `'vehicle'` | `"pedestrian"` |
| **`x`** | float | X coordinate in the world coordinate system (meters) | `12.5` |
| **`y`** | float | Y coordinate in the world coordinate system (meters) | `-3.4` |

**Tips**:

  * The sampling frequency of the data should match the model configuration parameter `args.fps` (default: 2.5 Hz). Otherwise, resample it first.
  * The coordinate system orientation must be consistent with the map data.

### 1.2 Map Data Format

The map should be wrapped as a `RasterizedMap` object containing raster data (a NumPy array) and physical boundary information.

```python
from src.dataset.base_dataset import RasterizedMap
import numpy as np

# 1. Prepare a rasterized map (0: traversable area/open space, 1: obstacle)
# A 2D array of shape (W, H)
grid_map = np.zeros((100, 100)) 

# 2. Define the physical-world boundaries corresponding to this map (unit: meters)
map_data = RasterizedMap(
    map=grid_map,
    xmin=0.0,  # x coordinate corresponding to the left edge of the map
    xmax=50.0, # x coordinate corresponding to the right edge of the map
    ymin=0.0,  # y coordinate corresponding to the bottom edge of the map
    ymax=50.0  # y coordinate corresponding to the top edge of the map
)
```

-----

## 2\. Processing Pipeline

If you provide the **raw data** described above, the system will convert it into model inputs through a chained processing pipeline. Understanding this process helps clarify what information the model actually uses and prepares you to switch to the advanced mode if needed.

The following logic is based on the implementation in `sample.py`:

1.  **Filtering**:
    Based on the current simulation start frame `frame_idx`, the system filters out the pedestrians (`ped_list`) and vehicles (`veh_list`) present at the current time.

2.  **Current position (`pos`)**:
    Directly extract the `(x, y)` coordinates of all pedestrians at frame `frame_idx`.

      * Result shape: `(#ped, 2)`

3.  **Current velocity (`vel`)**:
    Computed using finite differences: `(Pos_t - Pos_{t-1}) * FPS`.

      * Result shape: `(#ped, 2)`

4.  **Trajectory history (`hst`)**:
    Trace back and extract position data from the past `args.hist_step` frames (for example, the previous 8 frames).

      * **Note**: In the logic of `sample.py`, `hst` usually **does not include** the current frame `frame_idx`; it typically ends at `frame_idx - 1`.
      * Result shape: `(#ped, hist_step, 2)`

5.  **Vehicle trajectories (`veh`)**:
    Extract vehicle history trajectories. Unlike pedestrian history, vehicle history usually **includes** the current frame `frame_idx`.

      * Result shape: `(#veh, hist_step + 1, 2)`

6.  **Destination inference (`des`)**:
    By default, the system uses the position of each ID at its **last occurrence** in the DataFrame as its inferred destination.

      * Result shape: `(#ped, 2)`

7.  **Desired speed (`spd`)**:
    Compute the average movement-speed scalar over a future time window (for example, 5 seconds).

      * Result shape: `(#ped, 1)`

-----

## 3\. Advanced/Direct Mode

If you already have preprocessed data, or want to test specific hypotheses such as "What happens if the destination is somewhere else?", you can skip the DataFrame construction stage and feed tensors to the model directly.

Prepare the following variables as `torch.FloatTensor`s, and make sure they are placed on the same GPU/CPU device as the model.

**Dimension notation**:

  * `B`: Batch size (usually 1 or `sample_num`)
  * `N`: Number of pedestrians in the current scene
  * `M`: Number of vehicles in the current scene
  * `H`: History length (`args.hist_step`)

| Variable | Shape | Physical meaning and constraints | Customization suggestions |
| :--- | :--- | :--- | :--- |
| **`pos`** | `(B, N, 2)` | **Current positions** $(x, y)$. | Must match the map coordinate system exactly. |
| **`vel`** | `(B, N, 2)` | **Current velocities** $(v_x, v_y)$ in m/s. | Since this mode is fully customizable, you can modify this value to observe the model's response to different initial momentum. |
| **`hst`** | `(B, N, H, 2)` | **Pedestrian trajectory history**. Usually does not include the current frame. | If data is missing, you can fill it with the current position or use linear interpolation. |
| **`des`** | `(B, N, 2)` | **Destination coordinates**. | **This is the most commonly used control variable**. Changing it can guide the model to generate trajectories toward specific regions. |
| **`spd`** | `(B, N, 1)` | **Desired speed** (scalar). | Controls how urgently the agent moves. |
| **`veh`** | `(B, M, H+1, 2)` | **Vehicle trajectories**. Includes the current frame. | If there are no vehicles in the scene, this tensor can be empty or handled with a special padding strategy. |
| **`map`** | `(W, H)` | **Map feature map**. | 0 indicates free space and 1 indicates obstacles. The model maps it into physical space based on parameters such as `xmin/xmax`. |

### Example Invocation

```python
import torch
from src.model.model import Model

# 1. Initialize the model
model = Model(args)
# Load weights...

# 2. Prepare data (tensors)
# Assume you have already constructed tensors with the shapes described above
pos_tensor = ... 
vel_tensor = ...
# ...

# 3. Inject data into the model
# Note: you must set the map and vehicle inputs before setting pedestrian inputs
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

# 4. Compute local environmental features (must be called after set_map and set_ped)
model.set_sur_info() 

# 5. Start inference (for example, inside the diffusion loop)
output = model(
    noisy_acc=noisy_input, 
    denoise_t=t, 
    ped_length=ped_len, 
    veh_length=veh_len
)
```

In this way, you can completely bypass the data loader, flexibly control every input variable, and run highly customized simulation experiments.
