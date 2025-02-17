import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.envs.maze import MazeSingleAgentEnv


def test_maze() -> None:
    map_path: str = "tests/assets/board_maze.txt"

    env = MazeSingleAgentEnv(
        map_path=map_path, render_mode="human", max_steps=200, step_penalty_ratio=0
    )
    obs, _ = env.reset()
    env.render()

    while True:
        action = np.random.choice(list(env.actions_set))
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(f"episode ended after {env.step_count} steps")
            break
