import gymnasium as gym
import gym_multigrid
import numpy as np
import argparse


def run_save_the_city_game(env_id='SaveTheCity-v0', seed=None, max_episodes=1):
    """
    Runs the SaveTheCity environment with random ego agent actions and renders each step.

    Parameters
    ----------
    env_id : str
        Environment ID to run (e.g., 'SaveTheCity-v0', 'SaveTheCity-v1')
    seed : int | None
        Random seed for reproducibility
    max_episodes : int
        Number of episodes to run
    """

    # Create the environment
    env = gym.make(env_id, render_mode='human')

    # Get environment info
    total_agents = len(env.unwrapped.agents)
    num_ego_agents = len(env.unwrapped.ego_agent_indices)
    num_teammate_agents = len(env.unwrapped.non_ego_agent_indices)

    print(f"=" * 60)
    print(f"Running {env_id}")
    print(f"=" * 60)
    print(f"Total agents: {total_agents}")
    print(f"  - Ego agents (RL-controlled): {num_ego_agents}")
    print(f"  - Teammate agents (policy-controlled): {num_teammate_agents}")
    print(f"Grid size: {env.unwrapped.size}x{env.unwrapped.size}")
    print(f"Buildings: {env.unwrapped.num_buildings}")
    print(f"=" * 60)

    # Agent details
    for i, agent in enumerate(env.unwrapped.agents):
        agent_role = "EGO" if i in env.unwrapped.ego_agent_indices else "TEAMMATE"
        policy_name = agent.policy_name if agent.policy_name else "None (external control)"
        print(f"Agent {i}: {agent.agent_type.upper()} ({agent_role}) - Policy: {policy_name}")
    print(f"=" * 60)

    # Run episodes
    for episode in range(max_episodes):
        obs, info = env.reset(seed=seed)
        env.render()

        episode_over = False
        total_reward = 0
        step_count = 0

        print(f"\nEpisode {episode + 1} started...")

        while not episode_over:
            # Generate random actions ONLY for ego agents
            # Teammate agents generate their own actions internally
            ego_actions = [env.action_space.sample() for _ in range(num_ego_agents)]

            # Step environment (only provide ego actions)
            obs, reward, terminated, truncated, info = env.step(ego_actions)
            env.render()

            total_reward += reward
            step_count += 1

            # End episode if complete
            episode_over = terminated or truncated

        print(f"Episode {episode + 1} finished:")
        print(f"  - Steps: {step_count}")
        print(f"  - Total reward: {total_reward:.2f}")
        print(f"  - Terminated: {terminated} | Truncated: {truncated}")

        # Print per-agent stats
        for agent_key, stats in info.items():
            if agent_key.startswith("agent"):
                print(f"  - {agent_key}: {stats}")

    env.close()
    print(f"\n{'=' * 60}")
    print("Simulation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SaveTheCity environment")
    parser.add_argument(
        '--env',
        type=str,
        default='SaveTheCity-v0',
        choices=['SaveTheCity-v0', 'SaveTheCity-v1', 'SaveTheCity-v2', 'SaveTheCity-v3'],
        help='Environment version to run'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to run'
    )

    args = parser.parse_args()

    run_save_the_city_game(
        env_id=args.env,
        seed=args.seed,
        max_episodes=args.episodes
    )
