# CARLA Imitation Learning - PilotNet Framework

This repository provides an end-to-end framework for recording, training, and deploying Imitation Learning (Behavioral Cloning) models in the CARLA Simulator. It features a dual-architecture system capable of training standard **RGB images** or **Semantic Segmentation** matrices utilizing an enhanced version of NVIDIA's PilotNet.

## Core Features

- **Hybrid PilotNet Architecture (`PilotNetEnhancedConditional`)**: Extended CNN that parallelly assimilates visual features and dynamic vehicle speed to output a 3-dimensional physical vector: `[Throttle, Steering, Brake]`.
- **Multi-Channel Semantic Engine (`PilotNetSemanticConditional`)**: An advanced 5-Channel convolutional network that structurally ignores textures, weather, and lighting by transforming raw CARLA Class IDs into boolean layers.
- **Asymmetric Loss Weighting**: Custom MSE gradients highly penalizing lateral deviations (Steering weight: `0.7`) vs longitudinal errors (Throttle: `0.2`, Brake: `0.1`), forcing aggressive lane centering.
- **Online Data Augmentation**: Logical dataset tripling at RAM-level (x3 factor) applying `Flip` and `Rotations` dynamically. (Photometric noise is automatically disabled for semantic matrices to prevent categorical data corruption).

---

## Installation

Before running the Imitation Learning modules, you must install the specific neural network toolchain and Grad-CAM visualization utilities. Ensure your CARLA Simulator is up and running.

```bash
conda create -n carla_il python=3.10 -y
conda activate carla_il

conda install -c conda-forge numpy pandas matplotlib typing_extensions -y

pip install torch==2.10.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

pip install numpy pygame opencv-python grad-cam
```

> **Note:** As with the root repository, the `carla` Python package must be installed matching your local CARLA Simulator version (preferably via the `.egg` distribution).

---

## 1. Dataset Generation (`collect_dataset.py`)
This script acts as the automated data collector. It deploys an Ego Vehicle and an AI Safety Agent roaming indefinitely across the map avoiding dynamic obstacles, traffic lights, and stop signs.

### The "Dual-Pipeline" Semantic Extraction
For every frame (at $10$Hz), the `DatasetRecorder` concurrently dumps:
1. **RGB Cameras**: Standard lossy JPEG arrays.
2. **Semantic Masks**: Pure 1-Channel Mathematical 8-bit `PNGs`. 


**Usage:**
```bash
python collect_dataset.py --dataset ./dataset/ --traffic
```

---

## 2. Neural Network Training (`train.py`)
The training script manages dataset initialization, data augmentation, PyTorch hardware allocation, and model checkpointing.

### Input Types (`--image_type`)
- **`rgb`**: Loads standard 3-channel photos. Applies geometric (Rotation, Mirror) and photometric (Brightness, Gaussian Noise) augmentations.
- **`segsem`**: Loads 1-channel integers. Evaluates the pure class ID, and explodes it into a **5-Channel One-Hot Encoded (OHE) Tensor**:
  - **Channel 0 (RoadLines):** RoadLines (24)
  - **Channel 1 (Roads):** Roads (1)
  - **Channel 2 (Dynamics):** Bicycles/Signs/Fences/Guardrails/Poles/Lights (12, 13, 14, 15, 16, 18, 19)
  - **Channel 3 (Borders):** SideWalks/Fence/CustomGuardRail (2, 5, 28)
  - **Channel 4 (Pathing):** Pole/TrafficLight/TrafficSign (6, 7, 8)

### Usage Example:
```bash
# Train on pure RGB with Data Augmentation (x3 size)
python train.py --dataset_dir ./dataset/ --image_type rgb --data_aug --epochs 50

# Train on pure Mathematics (Semantic Modality)
python train.py --dataset_dir ./dataset/ --image_type segsem --data_aug --epochs 50
```
*Note: Checkpoints will be automatically saved as `il_best_pilotnet_rgb.pth` or `il_best_pilotnet_segsem.pth` preventing overwrites.*

---

## 3. Asynchronous Inference (`inference.py`)
Closes the loop by deploying the compiled `.pth` brain into the CARLA environment. 
- It actively strips the PID Controllers and automated Safety Agents. 
- The Neural Network gains **100% control** over the chassis `carla.VehicleControl`.
- Extracts the visual matrix dynamically, transforms it into PyTorch Tensors natively duplicating the exact `train.py` conditions (InterpolationMode `NEAREST` applied for Semantics).

**Usage:**
```bash
python inference.py --model_path il_best_pilotnet_segsem.pth --image_type segsem
```

*(Optional)* You can append the `--grad` flag to activate the real-time **Grad-CAM Attention Heatmap**, which will open a parallel visualization overlay on the left side of Pygame indicating exactly which physical pixels (e.g. road lines) the PilotNet is structurally focusing on to make its steering decisions.
