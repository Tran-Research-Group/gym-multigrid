import os
import pytest
import gymnasium as gym
import imageio

import gym_multigrid
from gym_multigrid.envs.prey_pred import (
    GreedyPredatorPolicy,
    GreedyPredatorActionOption,
)


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


def test_preypred_policy() -> None:
    animation_path = "tests/out/animations/test_preypred_policy.gif"
    env = gym.make("multigrid-preypred-v0")
    env.reset()
    frames = [env.render()]
    greedy_policy_1 = GreedyPredatorPolicy([1, 1, 1, 0], random_generator=env.np_random)
    greedy_policy_2 = GreedyPredatorPolicy([0, 0, 1, 1], random_generator=env.np_random)

    for _ in range(300):
        agent_1_pos = env.unwrapped.agents[1].pos
        agent_2_pos = env.unwrapped.agents[2].pos
        prey_agents = env.unwrapped.prey_agents
        grid = env.unwrapped.grid

        agent_1_option: GreedyPredatorActionOption = {
            "current_pos": agent_1_pos,
            "preys": prey_agents,
            "grid": grid,
        }
        agent_2_option: GreedyPredatorActionOption = {
            "current_pos": agent_2_pos,
            "preys": prey_agents,
            "grid": grid,
        }

        actions = [env.action_space.sample()] + [
            greedy_policy_1.act(agent_1_option),
            greedy_policy_2.act(agent_2_option),
        ]
        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())
        if terminated or truncated:
            break

    assert not truncated

    os.makedirs(os.path.dirname(animation_path), exist_ok=True)
    imageio.mimsave(animation_path, frames, duration=2, loop=20)
    assert os.path.exists(animation_path)
