import pytest
import gymnasium as gym
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_multigrid


@pytest.mark.parametrize("env_id", ["gym_multigrid:multigrid-collect-v0"])
def test_collect_game(env_id) -> None:
    """Test collect_game()"""
    env = gym.make(env_id)

    obs, _ = env.reset()
    while True:
        action = [env.action_space.sample()]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
