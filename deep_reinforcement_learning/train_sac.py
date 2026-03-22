"""
train_sac.py — SAC training script for CARLA lane-following.
Incorporates FrameStacking (4 frames) to provide temporal context.
"""
import argparse
import os
import math
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from CarlaEnv import CarlaEnv

class IdlePenaltyWrapper(gym.Wrapper):
    """
    Penaliza gradualmente al agente si mantiene una velocidad entre 0 y 5 km/h.
    Si llega a los 200 pasos (idle_steps) consecutivos a esa velocidad, el episodio termina con castigo alto.
    """
    def __init__(self, env, max_idle_steps=200):
        super().__init__(env)
        self.max_idle_steps = max_idle_steps
        self.idle_steps = 0

    def reset(self, **kwargs):
        self.idle_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        unwrapped = self.env.unwrapped
        if unwrapped.vehicle is not None and not terminated:
            vel = unwrapped.vehicle.get_velocity()
            spd_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            if spd_kmh < 5.0:
                self.idle_steps += 1
                if self.idle_steps <= self.max_idle_steps:
                    # Penalización progresiva: de 0.0 hasta -2.0 por frame
                    penalty = (self.idle_steps / self.max_idle_steps) * 2.0
                    reward -= penalty
                else:
                    # Timeout (Llegó a los 200 pasos estancado)
                    reward -= 50.0  # Castigo alto
                    terminated = True
                    print(f"[IdlePenaltyWrapper] Episodio terminado: Coche inactivo/bloqueado ({spd_kmh:.1f} km/h tras {self.max_idle_steps} steps constantes).")
            else:
                # Si supera los 5 km/h, reiniciamos el contador
                self.idle_steps = 0

        return obs, float(reward), terminated, truncated, info

def make_env(host, port, render):
    def _init():
        # Using default max_steps=2000, target_speed_kmh=60 (handled in CarlaEnv reward)
        env = CarlaEnv(host=host, port=port, render=render, max_steps=2000)
        env = IdlePenaltyWrapper(env, max_idle_steps=300)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="SAC training for CARLA lane-following")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=500_000, help="Total timesteps")
    parser.add_argument("--resume", type=str, default=None, help="Path to .zip checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs("checkpoints_sac", exist_ok=True)

    # 1. Create Environment
    base_env = DummyVecEnv([make_env(args.host, args.port, render=True)])
    
    # 2. Add FrameStack (4 frames)
    # This transforms observation from (4, 150, 200) to (16, 150, 200)
    env = VecFrameStack(base_env, n_stack=4, channels_order='first')

    # 3. Initialize SAC
    if args.resume:
        print(f"[SAC] Resuming from checkpoint: {args.resume}")
        model = SAC.load(args.resume, env=env)
    else:
        model = SAC(
            policy="CnnPolicy",
            env=env,
            buffer_size=15_000,      # Experience replay buffer size
            learning_starts=1_000,   # Steps before starting training
            batch_size=64,
            tau=0.005,               # Target network update rate
            gamma=0.99,
            train_freq=1,            # Update every step
            gradient_steps=1,
            ent_coef='auto',         # Automated entropy tuning (Learns optimal exploration)
            target_entropy='auto',
            policy_kwargs=dict(
                net_arch=[256, 128], # MLP heads after CNN
                features_extractor_kwargs=dict(features_dim=256),
                normalize_images=False, # Input is already [0, 1]
            ),
            verbose=1,
            device="auto"
        )

    callbacks = [
        CheckpointCallback(
            save_freq=10_000,
            save_path="./checkpoints_sac/",
            name_prefix="sac_carla",
        ),
    ]

    print(f"\n[SAC] Starting training for {args.steps:,} timesteps...")
    print(f"[SAC] Observation space: {env.observation_space}")
    print(f"[SAC] Action space:      {env.action_space}\n")

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[SAC] Training interrupted by user.")
    finally:
        model.save("sac_carla_lane")
        print("[SAC] Model saved to 'sac_carla_lane.zip'.")
        env.close()

if __name__ == "__main__":
    main()
