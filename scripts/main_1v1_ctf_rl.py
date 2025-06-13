from collections import deque
import os
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import torch
import imageio

from gym_multigrid.envs.ctf import Ctf1v1Env, GameStats


class GameStatsLogCallback(BaseCallback):
    def __init__(
        self,
        game_stats_keys: list[str] = [
            "defeated_blue_agents",
            "defeated_red_agents",
            "captured_blue_flags",
            "captured_red_flags",
        ],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.game_stats_keys: list[str] = game_stats_keys

    def _on_rollout_start(self) -> None:
        self.ep_game_stats_buffer: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        # Get the game_stats attribute from the environment
        infos: list[dict[str, Any]] = self.locals["infos"]
        for info in infos:
            self.ep_game_stats_buffer.append(info)

        return True

    def _on_rollout_end(self) -> None:
        # Get the game_stats attribute from the environment
        infos: list[dict[str, Any]] = self.ep_game_stats_buffer
        num_infos: str = len(infos)

        # Log average stats
        for key in self.game_stats_keys:
            self.logger.record(
                "game_stats/" + key, sum([s[key] for s in infos]) / num_infos
            )


total_timesteps: int = 1_000_000
tb_log_dir: str = "out/logs/ctf/"
tb_log_name: str = "ctf_1v1_ppo"
model_save_path: str = "out/models/ctf_ppo_1v1"
# Create the environment
map_path: str = "tests/assets/board_wall.txt"
env = Ctf1v1Env(
    map_path=map_path,
    render_mode="rgb_array",
    observation_option="tensor",
)

callback = GameStatsLogCallback()

print("GPU available: ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.exists(model_save_path + ".zip"):
    model = PPO.load(model_save_path, env=env, device=device)
else:
    # Create the RL model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tb_log_dir)

    # Train the model
    model.learn(
        total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=callback
    )

    # Save the model
    model.save(model_save_path)

# Save an animation
obs, _ = env.reset()

imgs = [env.render()]

while True:
    actions, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    imgs.append(env.render())

    if terminated or truncated:
        break

imageio.mimsave("out/animations/ctf_1v1_ppo.gif", imgs, fps=5)
