# Deep Reinforcement Learning with CARLA

This folder contains a complete Gymnasium-compatible environment and training pipelines to teach an autonomous agent how to drive in CARLA using Deep Reinforcement Learning (DRL). We utilize the robust [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) library for the underlying algorithms.

## Project Structure

* **`CarlaEnv.py`**: The core Gymnasium simulation wrapper. It connects to the CARLA server, spawns the ego-vehicle, processes the semantic segmentation camera into a mathematical tensor (4 binary channels), calculates the reward function (to enforce lane following, penalize collisions, etc.), and handles episode resets.
* **`train_ppo.py`**: Training script using the Proximal Policy Optimization (PPO) algorithm. Ideal for discrete and continuous control tasks balancing sample complexity and stability.
* **`train_sac.py`**: Training script using the Soft Actor-Critic (SAC) algorithm. An off-policy algorithm that maximizes continuous control rewards and entropy concurrently, usually yielding smoother mechanical driving logic.
* **`inference_ppo.py` / `inference_sac.py`**: Inference scripts to load the saved `.zip` model weights and test the fully trained AI inside the CARLA server in real-time.

## Installation

Before running the agents, you must install the specific reinforcement learning toolchains. Make sure your CARLA Simulator is running in the background.

```bash
# Install the core neural network engine and RL framework
pip install torch torchvision
pip install stable-baselines3 gymnasium

# Install utilities used by the environment
pip install numpy pygame opencv-python
```

> **Note:** Just like the rest of the repository, you must have the `carla` Python API installed strictly bound to the exact version of your simulator (preferably via the compiled `.egg` file).

## How to Run

1. Start your CARLA Simulator server (e.g., `./CarlaUE4.sh`).
2. Run the desired training script. For example, to train PPO:
   ```bash
   python train_ppo.py
   ```
   *The models will automatically save to disk periodically in `.zip` format.*
3. To evaluate your trained model visually, launch the inference script:
   ```bash
   python inference_ppo.py
   ```
