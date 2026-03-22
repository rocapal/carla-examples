"""
train_ppo.py — PPO training script for CARLA lane-following.

Usage:
    python train_ppo.py [--host 127.0.0.1] [--port 2000] [--steps 500000]

Requirements:
    pip install stable-baselines3 gymnasium opencv-python
"""
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from CarlaEnv import CarlaEnv


def make_env(host, port, render):
    def _init():
        return CarlaEnv(host=host, port=port, render=render, max_steps=2000)
    return _init


def main():
    parser = argparse.ArgumentParser(description="PPO training for CARLA lane-following")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--resume", type=str, default=None, help="Path to .zip checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    # Our obs is already (4, 66, 200) CHW — no VecTransposeImage needed
    env = DummyVecEnv([make_env(args.host, args.port, render=True)])

    # --- PPO with CnnPolicy ---
    if args.resume:
        print(f"[PPO] Resuming from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            # Core hyperparameters
            n_steps=1024,          # Steps per rollout per env before update
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.01,         # Entropy bonus for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            # CNN observation (4, 66, 200)
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 128], vf=[256, 128]),
                normalize_images=False,  # Already float32 in [0,1]
            ),
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=10_000,
            save_path="./checkpoints/",
            name_prefix="ppo_carla",
        ),
    ]

    print(f"\n[PPO] Starting training for {args.steps:,} timesteps...")
    print(f"[PPO] Observation space: {env.observation_space}")
    print(f"[PPO] Action space:      {env.action_space}\n")

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[PPO] Training interrupted by user.")
    finally:
        model.save("ppo_carla_semseg")
        print("[PPO] Model saved to 'ppo_carla_semseg.zip'.")
        env.close()


if __name__ == "__main__":
    main()
