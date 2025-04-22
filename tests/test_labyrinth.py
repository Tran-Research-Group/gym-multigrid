from typing import Literal
import pytest
import sys
import os

import numpy as np
from numpy.typing import NDArray
import imageio
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.envs.labyrinth import LabyrinthEnv, RewardConfig


def test_labyrinth() -> None:
    animation_dir: str = os.path.join("tests", "out", "animations")
    os.makedirs(animation_dir, exist_ok=True)
    animation_path: str = os.path.join(animation_dir, "labyrinth.gif")

    env = gym.make("independent_subtask_labyrinth-v0", max_episode_steps=10)

    obs, _ = env.reset()
    frames = [env.render()]
    terminated, truncated = False, False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")

    imageio.mimsave(animation_path, frames, loop=10)

