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
    total_reward = 0
    steps = 0

    while True:
        actions = {
            f"{a.index}": np.random.choice(list(env.actions)) for a in env.agents
        }
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward["0"]
        steps += 1
        if env.burnt_trees == 100:
            print(steps)
            break
        frames.append(env.render())
        if terminated or truncated:
            print(f"Total reward: {total_reward}")
            break

    # save_frames_as_gif(frames, path="./", filename="wildfire-", ep=0, fps=1, dpi=72)


test_wildfire()
