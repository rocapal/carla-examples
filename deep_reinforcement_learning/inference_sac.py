"""
inference_sac.py — Inference script to test the trained SAC agent in CARLA.
Must use VecFrameStack(n_stack=4) to match training conditions.
"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from CarlaEnv import CarlaEnv

def main():
    parser = argparse.ArgumentParser(description="Run SAC inference in CARLA")
    parser.add_argument("--model", type=str, required=True, help="Path to the .zip model")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--debug", action="store_true", help="Show matplotlib debug plot per step")
    args = parser.parse_args()

    print(f"\n[SAC Inference] Loading model: {args.model}")
    
    # 1. Initialize Base Environment (Async for smooth playback)
    def _init():
        return CarlaEnv(host=args.host, port=args.port, render=True, max_steps=5000, async_mode=True)
    
    base_env = DummyVecEnv([_init])
    
    # 2. Add FrameStack (CRITICAL: Must match training n_stack=4)
    env = VecFrameStack(base_env, n_stack=4, channels_order='first')

    if args.debug:
        plt.ion()
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        plt.tight_layout()

    try:
        # Load Model
        model = SAC.load(args.model, env=env)
        print("[SAC Inference] Model loaded successfully.")

        while True:
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            
            print(f"\n[SAC Inference] Starting new episode...")

            while not done:
                # SAC predict (deterministic=True)
                action, _states = model.predict(obs, deterministic=True)
                
                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                
                if args.debug:
                    # 'obs' is (1, 16, 150, 200). We show the latest 4-channel group (indices 12-15)
                    # Use base_env to get the single raw frame for display
                    raw_rgb = base_env.envs[0]._sem_raw
                    # Extract latest stack (the last 4 channels)
                    latest_channels = obs[0, 12:16] 
                    
                    axes[0,0].clear(); axes[0,0].imshow(raw_rgb); axes[0,0].set_title("Original Semantic")
                    axes[0,1].clear(); axes[0,1].imshow(latest_channels[0], cmap='gray'); axes[0,1].set_title("C0: RoadLines")
                    axes[0,2].clear(); axes[0,2].imshow(latest_channels[1], cmap='gray'); axes[0,2].set_title("C1: Roads")
                    axes[1,0].clear(); axes[1,0].imshow(latest_channels[2], cmap='gray'); axes[1,0].set_title("C2: Dynamics")
                    axes[1,1].clear(); axes[1,1].imshow(latest_channels[3], cmap='gray'); axes[1,1].set_title("C3: Borders")
                    plt.pause(0.001)

                total_reward += reward[0]
                steps += 1
                done = terminated[0] or truncated[0]

            print(f"[SAC Inference] Episode finished. Steps: {steps}, Total Reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\n[SAC Inference] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        print("[SAC Inference] Cleaning up...")
        env.close()

if __name__ == "__main__":
    main()
