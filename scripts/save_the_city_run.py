import gymnasium as gym
import gym_multigrid 
import numpy as np

def run_save_the_city_game():
    """
    Runs the SaveTheCity environment with random agent actions and renders each step.
    """
    
    # Create the custom environment (make sure it's registered!)
    env = gym.make('SaveTheCity-v0', render_mode='human')

    # Reset environment
    obs, info = env.reset()
    env.render()

    # Number of agents
    num_agents = len(env.unwrapped.agents)

    # Start simulation loop
    episode_over = False
    while not episode_over:
        # Sample random actions for each agent
        action = [env.action_space.sample() for _ in range(num_agents)]

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

        # End episode if complete
        episode_over = terminated or truncated

    env.close()


if __name__ == "__main__":
    run_save_the_city_game()
