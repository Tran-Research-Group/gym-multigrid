import pytest
import numpy as np
from gym_multigrid.envs.wildfire import WildfireEnv


def test_wildfire() -> None:
    env = WildfireEnv(max_steps=100, render_mode="rgb_array")
    obs, _ = env.reset()
    env.render()

    while True:
        action = np.random.choice(list(env.actions))
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
