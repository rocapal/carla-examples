# CARLA Autonomous Driving Examples

Welcome to the CARLA Examples repository! This project contains a collection of scripts and modules to learn, simulate, and train autonomous driving agents using the [CARLA Simulator](https://carla.org/).

The repository is structured into basic simulation examples and advanced machine learning training pipelines.

## Installation

To run the basic scripts (`01` to `06`), ensure you have the CARLA Simulator running in the background. The Python scripts themselves require a minimal set of libraries for tensor mathematics and window rendering.

You can install the necessary dependencies using `pip`:

```bash
pip install carla pygame numpy
```

## Basic Examples

These introductory scripts demonstrate how to interact with the CARLA Python API to spawn actors, control vehicles, and retrieve sensor data:

- **`01-HelloWorld.py`**: A minimal script to connect to the CARLA server, load a world, and spawn a basic ego-vehicle.
- **`02-TeleOperator.py`**: Allows manual control of a vehicle using the keyboard. It demonstrates how to listen for input events and apply mechanical control commands dynamically.
- **`03-Autopilot.py`**: Shows how to spawn multiple vehicles and pedestrians, handing over their control to CARLA's built-in Traffic Manager to simulate realistic traffic.
- **`04-Weather.py`**: Demonstrates how to programmatically alter environmental parameters like sun altitude, rain, and fog to test driving under different weather conditions.
- **`05-SensorSemantic.py`**: Attaches a semantic segmentation camera to a vehicle and processes the raw categorical arrays to visualize pixel-level scene understanding.
- **`06-SensorLidar.py`**: Attaches a LIDAR sensor to an ego-vehicle to capture, process, and display 3D point cloud data of the surroundings.

## Advanced Learning Modules

For more complex AI-driven vehicle control, this repository contains two dedicated folders exploring different automated driving paradigms. 
**Please navigate inside each folder to read their respective `README.md` files for detailed setup, architecture, and execution instructions.**

### 🧠 Deep Reinforcement Learning (`deep_reinforcement_learning/`)
Contains an environment wrapper compatible with Gymnasium to train self-driving agents using trial-and-error. It features implementations of state-of-the-art DRL algorithms such as PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic) to teach the car how to stay on the road while avoiding collisions.

### 🎥 Imitation Learning (`imitation_learning/`)
Contains dataset collection scripts and PyTorch architectures (such as PilotNet) to train a neural network via Behavioral Cloning. It records human or autopilot driving data (images + steering telemetry) and trains a vision-based model to predict vehicle controls directly from raw camera pixels.
