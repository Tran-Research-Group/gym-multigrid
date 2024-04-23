import pytest
import numpy as np
import gymnasium as gym
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.utils.misc import save_frames_as_gif


@pytest.mark.parametrize("env_id", ["gym_multigrid:wildfire-v0"])
def test_wildfire() -> None:
    env = gym.make(
        "wildfire-v0",
        size=35,
        num_agents=5,
        initial_fire_size=1,
        max_episode_steps=10,
        two_initial_fires=False,
        cooperative_reward=False,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[3, 3, 9, 9, 9],
        selfish_region_xmax=[10, 10, 16, 16, 16],
        selfish_region_ymin=[17, 17, 3, 3, 3],
        selfish_region_ymax=[22, 22, 13, 13, 13],
    )
    # trees_on_fire = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    # agent_pos = [(1, 1), (2, 2)]
    # state = env.construct_state(trees_on_fire, agent_pos)
    # env.reset(state=state)
    obs, _ = env.reset()
    frames = []
    frames.append(env.render())
    steps = 0
    num_episodes = 1
    # state = env.get_state()
    # env.get_state_interpretation(state)
    # np.set_printoptions(threshold=sys.maxsize)

    for _ in range(num_episodes):
        while True:
            actions = {
                f"{a.index}": np.random.choice(list(env.actions)) for a in env.agents
            }
            obs, reward, terminated, truncated, info = env.step(actions)
            steps += 1
            frames.append(env.render())
            # if steps == 5:
            #     env.reset(state=state)
            #     frames.append(env.render())
            if terminated or truncated:
                break

        # s = env.get_state()
        # env.interpretable_state(s)

        save_frames_as_gif(frames, path="./", filename="wildfire-", ep=0, fps=1, dpi=60)
    # save_frames_as_gif(frames, path="./", filename="wildfire-", ep=0, fps=3, dpi=30)
    # print(f"Average metric: {np.mean(ep_metric)}")


test_wildfire()
