import os
import pytest
import gymnasium as gym
import imageio

import gym_multigrid


def test_preypred_init() -> None:
    """Test PreyPredEnv initialization"""
    env = gym.make("multigrid-preypred-v0")
    assert env is not None


def test_preypred_step() -> None:
    """Test PreyPredEnv step"""
    animation_path = "tests/out/animations/test_preypred_step.gif"
    env = gym.make("multigrid-preypred-v0")
    env.reset()
    frames = [env.render()]

    for _ in range(300):
        actions = [env.action_space.sample() for _ in range(3)]
        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())
        if terminated or truncated:
            break

    assert not truncated

    os.makedirs(os.path.dirname(animation_path), exist_ok=True)
    imageio.mimsave(animation_path, frames, duration=2, loop=20)
    assert os.path.exists(animation_path)


def test_preypred_render() -> None:
    """Test PreyPredEnv render"""
    image_path = "tests/out/plots/test_preypred_render.png"
    env = gym.make("multigrid-preypred-v0")
    env.reset()
    img = env.render()

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imageio.imsave(image_path, img)
    assert os.path.exists(image_path)
