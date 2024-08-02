import sys

sys.path.append("../gym-multigrid")
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
    env = FourRooms(max_steps=10000, render_mode="human")
    obs = env.reset(seed=2)
    env.render()

    rew_sum = 0
    while True:
        action = np.random.choice(list(env.actions_set))
        # action = input()
        obs, reward, terminated, truncated, info = env.step(action)
        rew_sum += reward
        env.render()
        if terminated or truncated:
            break

    print(rew_sum)


# test_fr()
