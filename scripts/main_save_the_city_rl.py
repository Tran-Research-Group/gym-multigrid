"""
Training script for SaveTheCity environment using Stable Baselines3 PPO.

Configuration (SaveTheCity-v3):
    - 1 ego firefighter (RL-trained)
    - 2 teammate builders (using "builder" policy)
    - 1 teammate generalist (using "generalist" policy)
    - Grid: 15x15, 5 buildings, max 3000 steps per episode

Usage:
    poetry run python scripts/main_save_the_city_rl.py
"""

import os
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
import imageio
import gymnasium as gym
import numpy as np
import wandb

import gym_multigrid
from gym_multigrid.wrappers.single_agent import SingleAgentWrapper


class SaveTheCityStatsCallback(BaseCallback):
    """
    Callback to log SaveTheCity game statistics to TensorBoard and wandb.

    Logs per-episode stats (fires/buildings per agent, reward, length) at
    episode boundaries, and rollout-level aggregates at rollout boundaries.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_count: int = 0

    def _on_rollout_start(self) -> None:
        self.ep_stats_buffer: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals["infos"]
        dones: list[bool] = self.locals["dones"]

        for info in infos:
            self.ep_stats_buffer.append(info)

        # Log per-episode stats at episode boundaries
        for info, done in zip(infos, dones):
            if done:
                self._episode_count += 1
                ep_log: dict[str, Any] = {
                    "train/episode": self._episode_count,
                }
                total_fires = 0
                total_buildings = 0
                for key, val in info.items():
                    if key.startswith("agent") and isinstance(val, dict):
                        fires = val.get("fires_extinguished", 0)
                        buildings = val.get("buildings_completed", 0)
                        ep_log[f"train/{key}/fires_extinguished"] = fires
                        ep_log[f"train/{key}/buildings_completed"] = buildings
                        total_fires += fires
                        total_buildings += buildings
                ep_log["train/ep_fires_extinguished"] = total_fires
                ep_log["train/ep_buildings_completed"] = total_buildings
                wandb.log(ep_log, step=self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        if not self.ep_stats_buffer:
            return

        num_steps = len(self.ep_stats_buffer)

        total_fires = 0
        total_buildings = 0

        for stats in self.ep_stats_buffer:
            for key, val in stats.items():
                if key.startswith("agent") and isinstance(val, dict):
                    total_fires += val.get("fires_extinguished", 0)
                    total_buildings += val.get("buildings_completed", 0)

        fires_per_step = total_fires / num_steps
        buildings_per_step = total_buildings / num_steps

        self.logger.record(
            "game_stats/fires_extinguished_per_step", fires_per_step
        )
        self.logger.record(
            "game_stats/buildings_completed_per_step", buildings_per_step
        )

        log: dict[str, Any] = {
            "train/fires_extinguished_per_step": fires_per_step,
            "train/buildings_completed_per_step": buildings_per_step,
        }

        if self.model.ep_info_buffer:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            log["train/ep_rew_mean"] = np.mean(ep_rewards)
            log["train/ep_len_mean"] = np.mean(ep_lengths)

        wandb.log(log, step=self.num_timesteps)


# === Configuration ===
total_timesteps: int = 10_000
tb_log_dir: str = "out/logs/save_the_city/"
tb_log_name: str = "save_the_city_ppo_v3"
model_save_path: str = "out/models/save_the_city_ppo_v3"
animation_save_path: str = "out/animations/save_the_city_ppo_v3.gif"
wandb_project: str = "save-the-city"
wandb_run_name: str = "ppo-v3"

ppo_config: dict[str, Any] = {
    "env": "SaveTheCity-v3",
    "policy": "MlpPolicy",
    "total_timesteps": total_timesteps,
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

# Ensure output directories exist
os.makedirs("out/logs/save_the_city", exist_ok=True)
os.makedirs("out/models", exist_ok=True)
os.makedirs("out/animations", exist_ok=True)

# === Create environment ===
env = gym.make("SaveTheCity-v3", render_mode="rgb_array")
env = SingleAgentWrapper(env)

print("=" * 60)
print("SaveTheCity PPO Training")
print("=" * 60)
print(f"Environment: SaveTheCity-v3")
print(f"Grid: 15x15 | Buildings: 5 | Max steps: 3000")
print(f"Ego: 1 firefighter | Teammates: 1 builder")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print("=" * 60)

# === Initialize wandb ===
wandb.init(
    project=wandb_project,
    name=wandb_run_name,
    config=ppo_config,
)

callback = SaveTheCityStatsCallback()

# === GPU detection ===
print("GPU available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Train or load model ===
if os.path.exists(model_save_path + ".zip"):
    print(f"\nLoading existing model from {model_save_path}.zip")
    model = PPO.load(model_save_path, env=env, device=device)
else:
    model = PPO(
        ppo_config["policy"],
        env,
        verbose=0,
        tensorboard_log=tb_log_dir,
        device=device,
        learning_rate=ppo_config["learning_rate"],
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_range=ppo_config["clip_range"],
        ent_coef=ppo_config["ent_coef"],
    )

    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tb_log_name,
        callback=callback,
    )

    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}.zip")

# === Evaluation: run one episode and save GIF ===
print("\nRunning evaluation episode...")

obs, _ = env.reset()
imgs = [env.render()]

episode_reward = 0
step_count = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    imgs.append(env.render())

    episode_reward += reward
    step_count += 1

    if terminated or truncated:
        break

print(f"Evaluation: {step_count} steps, reward={episode_reward:.2f}")
print(f"Terminated: {terminated} | Truncated: {truncated}")

eval_log: dict[str, Any] = {
    "eval/episode_reward": episode_reward,
    "eval/episode_steps": step_count,
}
for agent_key, stats in info.items():
    if agent_key.startswith("agent") and isinstance(stats, dict):
        print(f"  {agent_key}: {stats}")
        for stat_key, stat_val in stats.items():
            eval_log[f"eval/{agent_key}/{stat_key}"] = stat_val

imageio.mimsave(animation_save_path, imgs, fps=5)
print(f"GIF saved to {animation_save_path}")

eval_log["eval/episode_video"] = wandb.Video(animation_save_path, fps=5, format="gif")
wandb.log(eval_log)

env.close()
wandb.finish()
