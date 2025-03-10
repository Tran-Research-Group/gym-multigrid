import sys
import os
import pytest
import numpy as np

import gymnasium as gym
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.envs.maze import MazeEnv


def test_maze_run() -> None:
    animation_save_path = "tests/out/animations/maze_animation.gif"

    env = gym.make("multigrid-maze-v0")

    obs, _ = env.reset()
    images = [env.render()]

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        images.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(animation_save_path, images, duration=0.5)
    assert os.path.exists(animation_save_path)
