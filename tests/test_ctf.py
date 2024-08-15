import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.envs.ctf import Ctf1v1Env


def test_ctf() -> None:
    map_path: str = "assets/board.txt"

    env = Ctf1v1Env(map_path=map_path, render_mode="human")
    obs, _ = env.reset()
    env.render()

    while True:
        action = np.random.choice(list(env.actions_set))
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break


test_ctf()
