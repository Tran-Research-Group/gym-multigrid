import pytest
import numpy as np
import gymnasium as gym
import sys
import os
import timeit
import cProfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.utils.misc import save_frames_as_gif


@pytest.mark.parametrize("env_id", ["gym_multigrid:wildfire-v0"])
def test_wildfire() -> None:
    env = gym.make(
        "wildfire-v0",
        size=5,
        num_agents=2,
        alpha=0.99,
        delta_beta=1,
        max_episode_steps=3,
        initial_fire_size=1,
        two_initial_fires=False,
        cooperative_reward=False,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[1, 3],
        selfish_region_xmax=[1, 3],
        selfish_region_ymin=[1, 1],
        selfish_region_ymax=[3, 3],
    )
    obs, _ = env.reset()
    frames = []
    frames.append(env.render())
    steps = 0
    num_episodes = 1
    # state = env.get_state()

    for _ in range(num_episodes):
        # start = timeit.default_timer()
        # print("the start time is: ", start)
        while True:
            actions = {
                f"{a.index}": np.random.choice(list(env.actions)) for a in env.agents
            }
            obs, reward, terminated, truncated, info = env.step(actions)
            steps += 1
            frames.append(env.render())
            if terminated or truncated:
                # env.reset(state=state)
                # print(env.get_state() == state)
                # print(env.selfish_region_trees_on_fire)
                break
        # print("the difference in time is: ", timeit.default_timer() - start)
        save_frames_as_gif(frames, path="./", filename="wildfire-", ep=0, fps=1, dpi=15)
    # save_frames_as_gif(frames, path="./", filename="wildfire-", ep=0, fps=3, dpi=30)
    # print(f"Average metric: {np.mean(ep_metric)}")


test_wildfire()
# cProfile.run("test_wildfire()", sort="cumtime")
