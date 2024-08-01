import pytest
import os

import numpy as np
import imageio
from stable_baselines3 import PPO

from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.policy.ctf.heuristic import (
    FightPolicy,
    CapturePolicy,
    PatrolPolicy,
    RwPolicy,
    PatrolFightPolicy,
)
from gym_multigrid.utils.map import load_text_map
import matplotlib.pyplot as plt


def test_fr() -> None:
    env = FourRooms(render_mode="human")
    obs = env.reset()
    env.render()

    while True:
        action = np.random.choice(list(env.actions_set))
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break


test_fr()
