import os
import pytest
import gymnasium as gym
import imageio

import gym_multigrid


def test_preypred_init() -> None:
    """Test PreyPredEnv initialization"""
    env = gym.make("multigrid-preypred-v0")
    assert env is not None


def test_preypred_render() -> None:
    """Test PreyPredEnv render"""
    image_path = "tests/out/plots/test_preypred_render.png"
    env = gym.make("multigrid-preypred-v0")
    env.reset()
    img = env.render()

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imageio.imsave(image_path, img)
    assert os.path.exists(image_path)
