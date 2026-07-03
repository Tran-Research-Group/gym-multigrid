import os

import gymnasium as gym
import imageio
from gym_multigrid.envs.lbf import GreedyPredatorPolicy


def test_lbf_init() -> None:
    """Test LBFEnv initialization"""
    env = gym.make("multigrid-lbf-v0")
    assert env is not None


def test_lbf_step() -> None:
    """Test LBFEnv step"""
    env = gym.make("multigrid-lbf-v0")
    env.reset()

    truncated: bool = False

    for _ in range(400):
        actions = [env.action_space.sample() for _ in range(3)]
        obs, reward, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    assert truncated


def test_lbf_render() -> None:
    """Test LBFEnv render"""
    image_path = "tests/out/plots/test_lbf_render.png"
    env = gym.make("multigrid-lbf-v0")
    env.reset()
    img = env.render()

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    imageio.imsave(image_path, img)
    assert os.path.exists(image_path)


def test_lbf_gif() -> None:
    """Test LBFEnv gif"""
    env = gym.make("multigrid-lbf-v0")
    env.reset()
    frames = [env.render()]

    truncated: bool = False

    for _ in range(400):
        actions = [env.action_space.sample() for _ in range(3)]
        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())
        if terminated or truncated:
            break

    os.makedirs("tests/out/animations", exist_ok=True)
    imageio.mimsave("tests/out/animations/lbf.gif", frames)
    assert os.path.exists("tests/out/animations/lbf.gif")


def test_lbf_heuristic() -> None:
    """Test LBFEnv heuristic"""
    env = gym.make("multigrid-lbf-v0")
    env.reset()
    frames = [env.render()]

    truncated: bool = False

    agent1_fruit = [
        [(5, 1), (5, 8)],
        [(15, 4), (15, 6)],
        [(13, 9), (14, 12), (13, 12), (13, 14)],
        [(5, 14), (5, 16), (6, 16)],
    ]
    agent2_fruit = [
        [(2, 2), (4, 6)],
        [(12, 4), (12, 3), (16, 5)],
        [(15, 11), (15, 15)],
        [(8, 11), (2, 11), (2, 14)],
    ]
    agent3_fruit = [
        [(2, 4), (2, 6)],
        [(10, 1), (11, 6)],
        [(12, 10), (12, 13), (12, 15), (14, 16)],
        [(7, 12), (6, 13), (3, 12), (3, 15), (3, 16), (5, 17)],
    ]
    agent1_goal = [(8, 5), (13, 8), (10, 14), (5, 10)]
    agent2_goal = [(8, 4), (14, 8), (10, 12), (3, 10)]
    agent3_goal = [(8, 3), (12, 8), (10, 13), (4, 10)]

    agent1 = GreedyPredatorPolicy(fruit_list=agent1_fruit, goal_list=agent1_goal)
    agent2 = GreedyPredatorPolicy(fruit_list=agent2_fruit, goal_list=agent2_goal)
    agent3 = GreedyPredatorPolicy(fruit_list=agent3_fruit, goal_list=agent3_goal)

    total_reward = 0
    for _ in range(1000):
        actions = [
            agent1.act(env.agents[0].pos, env.current_room, env.grid),
            agent2.act(env.agents[1].pos, env.current_room, env.grid),
            agent3.act(env.agents[2].pos, env.current_room, env.grid),
        ]

        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        frames.append(env.render())
        if terminated or truncated:
            break

    print(f"Total reward: {total_reward}")
    os.makedirs("tests/out/animations", exist_ok=True)
    imageio.mimsave("tests/out/animations/lbf_heuristic.gif", frames)
