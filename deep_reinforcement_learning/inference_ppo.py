import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from CarlaEnv import CarlaEnv

def main():
    parser = argparse.ArgumentParser(description="Run PPO inference in CARLA")
    parser.add_argument("--model", type=str, required=True, help="Path to the .zip model")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=20, help="Target display FPS")
    parser.add_argument("--debug", action="store_true", help="Show matplotlib debug plot per step")
    args = parser.parse_args()

    print(f"\n[Inference] Loading model: {args.model}")
    
    # Initialize Environment
    # Note: Use render=True for visualization and async_mode=True for smooth inference
    env = CarlaEnv(host=args.host, port=args.port, render=True, max_steps=5000, async_mode=True)

    if args.debug:
        plt.ion() # Interactive mode for live plotting
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        plt.tight_layout()

    try:
        # Load SB3 Model
        model = PPO.load(args.model, env=env)
        print("[Inference] Model loaded successfully.")

        while True:
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            print(f"\n[Inference] Starting new episode...")

            while not done:
                # Predict action using the trained policy (deterministic=True)
                action, _states = model.predict(obs, deterministic=True)
                
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                if args.debug:
                    # Update debug plot
                    # Row 0: Original | C0 | C1
                    # Row 1: C2 | C3 | (empty)
                    axes[0,0].clear(); axes[0,0].imshow(env._sem_raw); axes[0,0].set_title("Original Semantic")
                    axes[0,1].clear(); axes[0,1].imshow(obs[0], cmap='gray'); axes[0,1].set_title("C0: RoadLines")
                    axes[0,2].clear(); axes[0,2].imshow(obs[1], cmap='gray'); axes[0,2].set_title("C1: Roads")
                    axes[1,0].clear(); axes[1,0].imshow(obs[2], cmap='gray'); axes[1,0].set_title("C2: Dynamics")
                    axes[1,1].clear(); axes[1,1].imshow(obs[3], cmap='gray'); axes[1,1].set_title("C3: Borders")
                    plt.pause(0.001)
                
                total_reward += reward
                steps += 1
                done = terminated or truncated

                # Minimal sleep to match target FPS if needed (CARLA sync mode usually handles this)
                # time.sleep(1.0 / args.fps)

            print(f"[Inference] Episode finished. Steps: {steps}, Total Reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\n[Inference] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        print("[Inference] Cleaning up...")
        env.close()

if __name__ == "__main__":
    main()
