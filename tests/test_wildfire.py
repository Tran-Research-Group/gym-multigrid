import pytest
import numpy as np
import gymnasium as gym
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.utils.misc import save_frames_as_gif


@pytest.mark.parametrize("env_id", ["gym_multigrid:wildfire-v0"])
def test_wildfire() -> None:
    env = gym.make("wildfire-v0", max_episode_steps=100)
    obs, _ = env.reset()
    frames = []
    frames.append(env.render())

    while True:
        actions = [
            np.random.choice(list(env.actions)),
            np.random.choice(list(env.actions)),
        ]
        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())
        if terminated or truncated:
            break
    save_frames_as_gif(frames, path="./", filename="wildfire-", ep=0)


test_wildfire()
