import gymnasium as gym  # Import Gymnasium (a maintained fork of OpenAI Gym)
import gym_multigrid  # Import the gym-multigrid package (registers the environment)


def run_collect_game():
    """
    Runs the 'multigrid-collect-v1' environment with multiple agents in human mode.

    - The environment consists of multiple agents collecting objects in a grid world.
    - Each agent takes random actions at each step.
    - The game runs until all agents reach a termination condition.
    - The environment is rendered in real-time.

    Returns:
        None
    """
    
    # Create the environment in human rendering mode
    env = gym.make('multigrid-collect-v1', render_mode='human')

    # Reset the environment and retrieve the initial observation and info
    observation, info = env.reset()
    env.render()  # Render the initial state

    # Access the number of agents in the environment
    num_agents = len(env.unwrapped.agents_index)  

    # Initialize episode status
    episode_over = False

    while not episode_over:
        # Generate a random action for each agent
        action = [env.action_space.sample() for _ in range(num_agents)]
        
        # Execute the chosen actions and retrieve updated observations and rewards
        observation, rewards, terminated, truncated, info = env.step(action)

        # Render the updated environment state
        env.render()

        # Update termination status: Episode ends if all agents are terminated or truncated
        episode_over = terminated or truncated  

    # Close the environment properly after the episode ends
    env.close()


# Run the game
if __name__ == "__main__":
    run_collect_game()
