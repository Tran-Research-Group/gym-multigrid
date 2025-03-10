import sys
import os
import pytest
import numpy as np

import gymnasium as gym
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.envs.maze import LayoutConfig, MazeEnv


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


def test_maze_reset() -> None:
    env = gym.make("multigrid-maze-v0")
    obs, _ = env.reset()
    assert obs.shape == (2, 10, 10)
    assert obs[0, 9, 9] == 2
    assert obs[1, 5, 5] == 1


def test_maze_reset_options() -> None:
    layout_config = {
        "init_agent_positions": [(3, 6)],
        "flag_positions": [(7, 9)],
    }
    options = {
        "layout_config": layout_config,
    }
    env = gym.make("multigrid-maze-v0")

    obs, _ = env.reset(options=options)
    assert obs.shape == (2, 10, 10)
    assert obs[0, 7, 9] == 2
    assert obs[1, 3, 6] == 1
    assert obs[0, 9, 9] == 0
    assert obs[1, 5, 5] == 0
