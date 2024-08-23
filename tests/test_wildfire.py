import sys
import os
import pytest
import numpy as np
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.utils.misc import save_frames_as_gif


def test_wildfire() -> None:
    """Function to test the environment's functionality. Runs episodes with random agents in the Wildfire environment and save episode renders as GIFs."""
    env = gym.make(
        "wildfire-v0",
        alpha=0.3,
        beta=0.8,
        max_episode_steps=1,
        num_agents=2,
        agent_start_positions=((4, 6), (10, 12)),
        size=17,
        initial_fire_size=3,
        cooperative_reward=False,
        render_selfish_region_boundaries=True,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[2, 8],
        selfish_region_xmax=[6, 12],
        selfish_region_ymin=[4, 10],
        selfish_region_ymax=[8, 14],
    )
    obs, _ = env.reset()
    frames = []
    frames.append(env.render())
    steps = 0
    num_episodes = 1

    for ep in range(num_episodes):
        while True:
            actions = {
                f"{a.index}": np.random.choice(list(env.actions)) for a in env.agents
            }
            obs, reward, terminated, truncated, _ = env.step(actions)
            steps += 1
            frames.append(env.render())
            if terminated or truncated:
                break
        save_frames_as_gif(
            frames, path="./", filename="wildfire", ep=ep, fps=0.1, dpi=20
        )


test_wildfire()
