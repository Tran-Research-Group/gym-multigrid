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
    steps = 0
    ep_metric = []  # store the metric for each episode
    num_episodes = 1000

    for _ in range(num_episodes):
        while True:
            actions = {
                f"{a.index}": np.random.choice(list(env.actions)) for a in env.agents
            }
            obs, reward, terminated, truncated, info = env.step(actions)
            steps += 1
            frames.append(env.render())
            if terminated or truncated:
                ep_metric.append(
                    1
                    - (env.trees_on_fire + info["0"]["burnt trees"])
                    / env.grid_size_without_walls**2,
                )
                break

        # save_frames_as_gif(
        #     frames, path="./", filename="wildfire-", ep=0, fps=10, dpi=32
        # )

    print(f"Average metric: {np.mean(ep_metric)}")


test_wildfire()
