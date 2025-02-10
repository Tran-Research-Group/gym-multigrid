import os
import pprint

import gymnasium as gym
import imageio

import gym_multigrid

animation_path = "out/animations/test_preypred_step.gif"
animation_path_2 = "out/animations/test_preypred_step_2.gif"

env = gym.make("multigrid-preypred-v0")

obs, _ = env.reset()
frames = [env.render()]

while True:
    actions = [env.action_space.sample() for _ in range(3)]
    obs, reward, terminated, truncated, _ = env.step(actions)
    print(f"Reward: {reward}")
    pprint.pprint(obs)
    frames.append(env.render())
    if terminated or truncated:
        break

os.makedirs(os.path.dirname(animation_path), exist_ok=True)
imageio.mimsave(animation_path, frames, duration=2, loop=20)

obs, _ = env.reset()
frames = [env.render()]

while True:
    actions = [env.action_space.sample() for _ in range(3)]
    obs, reward, terminated, truncated, _ = env.step(actions)
    print(f"Reward: {reward}")
    pprint.pprint(obs)
    frames.append(env.render())
    if terminated or truncated:
        break

os.makedirs(os.path.dirname(animation_path_2), exist_ok=True)
imageio.mimsave(animation_path_2, frames, duration=2, loop=20)
