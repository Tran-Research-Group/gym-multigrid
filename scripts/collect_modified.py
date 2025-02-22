import gymnasium as gym
import gym_multigrid

env = gym.make('multigrid-collect-v1', render_mode='human')
observation, info = env.reset()
env.render()
# Access the number of agents correctly
num_agents = len(env.unwrapped.agents_index)
episode_over = False
while not episode_over:
    action = [env.action_space.sample() for _ in range(num_agents)]
    observation, rewards, terminated, truncated, info = env.step(action)
    env.render()
    episode_over = terminated or truncated

env.close()

# print(gym.envs.registry.keys())
