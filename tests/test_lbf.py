import os

import gymnasium as gym
import imageio
import numpy as np
from gym_multigrid.envs.lbf import LBFGameEnv

def test_lbf_init() -> None:
    """Test LBFEnv initialization"""
    env = gym.make("multigrid-lbf-v0")
    assert env is not None

def test_lbf_step() -> None:
    """Test LBFEnv step"""
    env = gym.make("multigrid-lbf-v0")
    env.reset()
    frames = [env.render()]

    truncated: bool = False

    for _ in range(400):
        actions = [env.action_space.sample() for _ in range(3)]
        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())
        if terminated or truncated:
            break

    assert truncated


def test_lbf_render() -> None:
    """Test LBFEnv render"""
    image_path = "tests/out/plots/test_lbf_render.png"
    env = gym.make("multigrid-lbf-v0")
    env.reset()
    img = env.render()

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imageio.imsave(image_path, img)
    assert os.path.exists(image_path)
