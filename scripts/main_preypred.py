import os
import pprint

import gymnasium as gym
import imageio

import gym_multigrid
from gym_multigrid.envs.prey_pred import (
    GreedyPredatorPolicy,
    GreedyPredatorActionOption,
)

animation_path = "out/animations/test_preypred_step.gif"

env = gym.make("multigrid-preypred-v0")

obs, _ = env.reset()
frames = [env.render()]

# Initialize greedy policies for heuristic predators
greedy_policy_1 = GreedyPredatorPolicy([1, 1, 1, 0], random_generator=env.np_random)
greedy_policy_2 = GreedyPredatorPolicy([0, 0, 1, 1], random_generator=env.np_random)

while True:
    # Create inputs for the greedy policies
    prey_agents = env.unwrapped.prey_agents
    grid = env.unwrapped.grid
    agent_1_option: GreedyPredatorActionOption = {
        "current_pos": env.unwrapped.agents[1].pos,
        "preys": prey_agents,
        "grid": grid,
    }
    agent_2_option: GreedyPredatorActionOption = {
        "current_pos": env.unwrapped.agents[2].pos,
        "preys": prey_agents,
        "grid": grid,
    }

    # Get actions from the greedy policies
    # Agent 0 is a random agent
    actions = [env.action_space.sample()] + [
        greedy_policy_1.act(agent_1_option),
        greedy_policy_2.act(agent_2_option),
    ]
    actions = [env.action_space.sample() for _ in range(3)]
    obs, reward, terminated, truncated, _ = env.step(actions)
    print(f"Reward: {reward}")
    pprint.pprint(obs)
    frames.append(env.render())
    if terminated or truncated:
        break

os.makedirs(os.path.dirname(animation_path), exist_ok=True)
imageio.mimsave(animation_path, frames, duration=2, loop=20)

# obs, _ = env.reset()
# frames = [env.render()]

# while True:
#     actions = [env.action_space.sample() for _ in range(3)]
#     obs, reward, terminated, truncated, _ = env.step(actions)
#     print(f"Reward: {reward}")
#     pprint.pprint(obs)
#     frames.append(env.render())
#     if terminated or truncated:
#         break

# os.makedirs(os.path.dirname(animation_path_2), exist_ok=True)
# imageio.mimsave(animation_path_2, frames, duration=2, loop=20)
