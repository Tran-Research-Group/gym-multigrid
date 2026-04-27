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
from gym_multigrid.envs.save_the_city import SaveTheCityActions
from gym_multigrid.wrappers.single_agent import SingleAgentWrapper

_ACTION_NAMES = [a.name.lower() for a in SaveTheCityActions]


class SaveTheCityStatsCallback(BaseCallback):
    """
    Callback to log SaveTheCity game statistics to TensorBoard and wandb.

    Per-episode (at episode boundary):
        - Reward total and breakdown (extinguish / build / burn penalty)
        - Fires extinguished, buildings completed, buildings burned
        - Action distribution (fraction of each action taken)
        - Whether episode terminated naturally or was truncated

    Per-rollout:
        - Fires/buildings per step
        - Reward stats (mean, min, max, std) over completed episodes
        - PPO training metrics (policy loss, value loss, entropy, KL, etc.)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_count: int = 0
        self._ep_reward: float = 0.0
        self._ep_length: int = 0
        self._ep_actions: list[int] = [0] * len(SaveTheCityActions)

    def _on_rollout_start(self) -> None:
        self.ep_stats_buffer: list[dict[str, Any]] = []
        self._rollout_rewards: list[float] = []
        self._rollout_lengths: list[int] = []

    def _on_step(self) -> bool:
        info: dict[str, Any] = self.locals["infos"][0]
        done: bool = self.locals["dones"][0]
        reward: float = float(self.locals["rewards"][0])
        action: int = int(self.locals["actions"][0])

        self._ep_reward += reward
        self._ep_length += 1
        self._ep_actions[action] += 1
        self.ep_stats_buffer.append(info)

        if done:
            self._episode_count += 1

            total_fires = 0
            total_buildings = 0
            for key, val in info.items():
                if key.startswith("agent") and isinstance(val, dict):
                    total_fires += val.get("fires_extinguished", 0)
                    total_buildings += val.get("buildings_completed", 0)

            buildings_burned: int = info.get("buildings_burned", 0)
            truncated: bool = info.get("TimeLimit.truncated", False)

            reward_from_extinguish = total_fires * 50
            reward_from_builds = total_buildings * 100
            penalty_from_burns = buildings_burned * 100

            ep_log: dict[str, Any] = {
                "train/episode": self._episode_count,
                # Reward
                "train/ep_reward": self._ep_reward,
                "train/ep_reward_from_extinguish": reward_from_extinguish,
                "train/ep_reward_from_builds": reward_from_builds,
                "train/ep_penalty_from_burns": -penalty_from_burns,
                # Game outcomes
                "train/ep_fires_extinguished": total_fires,
                "train/ep_buildings_completed": total_buildings,
                "train/ep_buildings_burned": buildings_burned,
                "train/ep_length": self._ep_length,
                # 1 = terminated naturally, 0 = hit max steps
                "train/ep_terminated": 0 if truncated else 1,
            }

            # Per-agent breakdowns
            for key, val in info.items():
                if key.startswith("agent") and isinstance(val, dict):
                    ep_log[f"train/{key}/fires_extinguished"] = val.get("fires_extinguished", 0)
                    ep_log[f"train/{key}/buildings_completed"] = val.get("buildings_completed", 0)

            # Action distribution
            for i, name in enumerate(_ACTION_NAMES):
                ep_log[f"train/action_frac/{name}"] = self._ep_actions[i] / max(self._ep_length, 1)

            wandb.log(ep_log, step=self.num_timesteps)

            self._rollout_rewards.append(self._ep_reward)
            self._rollout_lengths.append(self._ep_length)

            # Reset episode accumulators
            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_actions = [0] * len(SaveTheCityActions)

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

        self.logger.record("game_stats/fires_extinguished_per_step", fires_per_step)
        self.logger.record("game_stats/buildings_completed_per_step", buildings_per_step)

        rollout_log: dict[str, Any] = {
            "train/fires_extinguished_per_step": fires_per_step,
            "train/buildings_completed_per_step": buildings_per_step,
        }

        if self._rollout_rewards:
            rollout_log["train/ep_rew_mean"] = float(np.mean(self._rollout_rewards))
            rollout_log["train/ep_rew_min"] = float(np.min(self._rollout_rewards))
            rollout_log["train/ep_rew_max"] = float(np.max(self._rollout_rewards))
            rollout_log["train/ep_rew_std"] = float(np.std(self._rollout_rewards))
            rollout_log["train/ep_len_mean"] = float(np.mean(self._rollout_lengths))

        # PPO training metrics from the previous rollout's update
        ppo_metric_keys = [
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/explained_variance",
            "train/loss",
        ]
        for key in ppo_metric_keys:
            val = self.logger.name_to_value.get(key)
            if val is not None:
                rollout_log[f"ppo/{key.split('/')[-1]}"] = val

        wandb.log(rollout_log, step=self.num_timesteps)


# === Configuration ===
total_timesteps: int = 2_000_000
tb_log_dir: str = "out/logs/save_the_city/"
tb_log_name: str = "save_the_city_ppo_v3"
model_save_path: str = "out/models/save_the_city_ppo_v3"
animation_save_path: str = "out/animations/save_the_city_ppo_v3.gif"
wandb_project: str = "save-the-city"
wandb_run_name: str = "ppo-v3-updated-2M"

ppo_config: dict[str, Any] = {
    "env": "SaveTheCity-v3",
    "policy": "MlpPolicy",
    "total_timesteps": total_timesteps,
    "learning_rate": 3e-4,
    "n_steps": 2048,
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
print(f"Grid: 10x10 | Buildings: 5 | Max steps: 500")
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
