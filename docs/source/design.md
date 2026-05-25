# Crowd Simulation Project - Design and Architecture Document

## 1\. Project Overview

This project aims to achieve high-fidelity multi-agent pedestrian trajectory simulation based on diffusion models. The system takes scene maps and pedestrian trajectory history as input, predicts future crowd motion trends, and visualizes them on the web. The project consists of three major modules: model training, inference sampling, and web-based interactive visualization.

## 2\. System Architecture

### 2.1 Core Module Breakdown

  * **Dataset (`src.dataset`)**: Responsible for standardized loading of multi-source heterogeneous data. Supports mainstream datasets such as ETH, UCY, SDD, GC, WayMo, and ORCA. The core class `BaseDataset` implements unified sliding-window sampling, coordinate normalization, and data caching mechanisms.
  * **Model (`src.model`)**:
      * `Model`: Base Transformer model with attention mechanisms over pedestrians, vehicles, and maps.
      * `RelativeModel`: Improved model using relative-coordinate encoding, with better generalization.
      * `NewModel`: Latest experimental model architecture.
  * **Diffusion (`src.diffusion`)**: Implements two sampling strategies, DDPM (probabilistic diffusion) and DDIM (implicit diffusion), to progressively restore Gaussian noise into physically plausible trajectory accelerations.
  * **Web (`src.web`)**: Visualization backend based on FastAPI and WebSockets, supporting real-time streaming of simulation results to the frontend.

### 2.2 Data Flow

1.  **Input**: Raw trajectory CSV/Txt files plus scene images.
2.  **Preprocessing**: `BaseDataset` performs coordinate transformation (homography), resampling, and normalization.
3.  **Training**: `train.py` trains the diffusion model using sliding-window data to predict acceleration over future `pred_step` frames.
4.  **Inference**: `sample.py` or `simulate.py` uses the trained model together with Social Force Guidance for multi-frame autoregressive rollout.
5.  **Visualization**: Results are sent to the frontend through WebSocket and rendered as trajectories on the map using Plotly.js.

## 3\. Key Algorithm Design

### 3.1 Diffusion Process

**DDIM** is used for fast sampling. The model predicts **acceleration** rather than position directly, which helps ensure smoothness and physical plausibility in the generated trajectories.

### 3.2 Guidance Strategy

To improve controllability, the denoising process introduces Classifier-Free Guidance (CFG) and gradient-based energy guidance:

  * **Destination guidance**: Guides pedestrians toward preset target locations.
  * **Obstacle-avoidance guidance**: Uses `get_force_map` to compute a map potential field that repels pedestrians away from obstacles.

## 4\. API Interface

### 4.1 Training Interface

```bash
python train.py --name [experiment_name] --datasets [dataset_list] --loss_type [noise|accelerate]
```

### 4.2 Web Simulation Interface

The web backend listens on `ws://0.0.0.0:12345/ws`.

  * **Start Action**: `{ "action": "start", "dataset_name": "ETH", "frame_idx": 100, "frame_num": 200 }`
  * **Response**: Contains the streamed pedestrian IDs, types, and `(x, y)` coordinates for each frame.
