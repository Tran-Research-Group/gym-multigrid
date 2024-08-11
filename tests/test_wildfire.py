import sys
import os
import pytest
import numpy as np
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.utils.misc import save_frames_as_gif


@pytest.mark.parametrize("env_id", ["gym_multigrid:wildfire-v0"])
def test_wildfire() -> None:
    """Runs episodes in the Wildfire environment. Can be used to test the environment's functionality."""
    env = gym.make(
        "wildfire-v0",
        alpha=0.99,
        delta_beta=1,
        max_episode_steps=30,
        num_agents=4,
        agent_start_positions=((1, 1), (2, 2), (13, 13), (14, 14)),
        agent_groups=((0, 1), (2, 3)),
        size=17,
        initial_fire_size=3,
        two_initial_fires=False,
        cooperative_reward=False,
        render_selfish_region_boundaries=True,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[1, 5],
        selfish_region_xmax=[3, 7],
        selfish_region_ymin=[1, 10],
        selfish_region_ymax=[3, 15],
    )
    obs, _ = env.reset()
    frames = []
    frames.append(env.render())
    steps = 0
    num_episodes = 1
    state = env.get_state()
    trees_on_fire, _ = env.get_state_interpretation(state, print_interpretation=False)
    print(
        np.alltrue(
            env.construct_state(trees_on_fire, env.agent_start_positions, 0) == state
        )
    )

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
        save_frames_as_gif(frames, path="./", filename="wildfire", ep=ep, fps=1, dpi=30)


test_wildfire()
