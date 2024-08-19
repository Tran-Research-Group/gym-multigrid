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
        alpha=0.2,
        beta=0.9,
        delta_beta=0.5,
        max_episode_steps=100,
        num_agents=8,
        agent_groups=[(0, 1, 2, 3), (4, 5, 6, 7)],
        agent_start_positions=(
            (9, 31),
            (10, 31),
            (9, 32),
            (10, 32),
            (28, 11),
            (29, 11),
            (28, 12),
            (29, 12),
        ),
        size=42,
        initial_fire_size=5,
        two_initial_fires=False,
        cooperative_reward=False,
        render_selfish_region_boundaries=True,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[5, 26],
        selfish_region_xmax=[14, 31],
        selfish_region_ymin=[27, 9],
        selfish_region_ymax=[36, 14],
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
            frames, path="./", filename="wildfire", ep=ep, fps=15, dpi=120
        )


test_wildfire()
