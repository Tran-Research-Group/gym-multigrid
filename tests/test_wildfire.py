import pytest
import numpy as np
import gymnasium as gym
import sys
import os
import timeit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.utils.misc import save_frames_as_gif


@pytest.mark.parametrize("env_id", ["gym_multigrid:wildfire-v0"])
def test_wildfire() -> None:
    env = gym.make(
        "wildfire-v0",
        size=17,
        num_agents=2,
        initial_fire_size=3,
        max_episode_steps=100,
        two_initial_fires=False,
        cooperative_reward=True,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[3, 3],
        selfish_region_xmax=[10, 10],
        selfish_region_ymin=[7, 7],
        selfish_region_ymax=[12, 12],
    )
    obs, _ = env.reset()
    # frames = []
    # frames.append(env.render())
    steps = 0
    num_episodes = 1
    state = env.get_state()

    for _ in range(num_episodes):
        start = timeit.default_timer()
        print("the start time is: ", start)
        while True:
            actions = {
                f"{a.index}": np.random.choice(list(env.actions)) for a in env.agents
            }
            obs, reward, terminated, truncated, info = env.step(actions)
            steps += 1
            # frames.append(env.render())
            if terminated or truncated:
                break
        print("the difference in time is: ", timeit.default_timer() - start)
        # save_frames_as_gif(frames, path="./", filename="wildfire-", ep=0, fps=1, dpi=60)
    # save_frames_as_gif(frames, path="./", filename="wildfire-", ep=0, fps=3, dpi=30)
    # print(f"Average metric: {np.mean(ep_metric)}")


test_wildfire()
