from typing import Literal
import pytest
import sys
import os

import numpy as np
from numpy.typing import NDArray
import imageio
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gym_multigrid.envs.labyrinth import LabyrinthEnv, RewardConfig


def test_labyrinth() -> None:

    animation_path: str = "tests/out/animations/labyrinth.gif"

    env = gym.make("multigrid-labyrinth-v0")

    obs, _ = env.reset()
    frames = [env.render()]

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        if terminated or truncated:
            break

    imageio.mimsave(animation_path, frames, loop=10)


@pytest.mark.parametrize(
    "observation_option, expected_shape",
    [
        ("final_goal", (2 * 2,)),
        ("intermediate_goal", (5 * 2,)),
    ],
)
def test_observation_space(
    observation_option: Literal["final_goal", "intermediate_goal"],
    expected_shape: tuple[int, int],
) -> None:
    env = LabyrinthEnv(observation_option=observation_option)
    obs: dict[str, NDArray[np.int_]] = env.reset()[0]
    for key in obs.keys():
        assert obs[key].shape == expected_shape


@pytest.mark.parametrize(
    "init_pos, door_id",
    [
        (((3, 1), (3, 2), (3, 3)), 0),
        (((2, 5), (2, 6), (2, 7)), 1),
        (((5, 3), (5, 4), (5, 5)), 2),
    ],
)
def test_check_doors(init_pos: tuple[tuple[int, int], ...], door_id: int) -> None:

    animation_path: str = f"tests/out/animations/labyrinth_door_{door_id}.gif"

    env = LabyrinthEnv(init_pos=init_pos)
    action = [env.actions.right, env.actions.right, env.actions.right]
    obs, _ = env.reset()
    frames = [env.render()]

    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        if terminated or truncated:
            break

    imageio.mimsave(animation_path, frames, loop=10)


def test_find_first_goal_groups() -> None:

    env = LabyrinthEnv()
    subtask_ids: list[int] = env._find_first_subtasks()

    assert subtask_ids == [0, 1]


def test_compute_reward_final_goal_agent_on_goal():
    reward_config: RewardConfig = {
        "reward_option": "final_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }

    init_pos: tuple[tuple[int, int], ...] = ((7, 3), (7, 4), (7, 5))
    p_intended_action: float = 1
    env = LabyrinthEnv(
        p_intended_action=p_intended_action,
        reward_config=reward_config,
        init_pos=init_pos,
    )

    action = [env.actions.right, env.actions.right, env.actions.right]

    obs, _ = env.reset()
    env.current_subtasks = [env.subtask_dict[3]]
    env.rewarded_subtasks = [env.subtask_dict[3]]
    rewards: list[int] = []
    frames = [env.render()]

    for i in range(1):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(
        "tests/out/animations/labyrinth_compute_reward_final_goal_agent_on_goal.gif",
        frames,
        loop=10,
    )

    assert rewards == [
        reward_config["movement_reward"] * 3
        + reward_config["agent_on_goal_reward"] * 3
        + reward_config["all_agents_on_goal_reward"]
    ]
    assert terminated


def test_compute_reward_intermediate_goal_agent_on_goal_2_right():
    reward_config: RewardConfig = {
        "reward_option": "intermediate_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }

    init_pos: tuple[tuple[int, int], ...] = ((5, 3), (5, 4), (5, 5))
    p_intended_action: float = 1
    env = LabyrinthEnv(
        p_intended_action=p_intended_action,
        reward_config=reward_config,
        init_pos=init_pos,
    )

    action = [env.actions.right, env.actions.right, env.actions.right]

    obs, _ = env.reset()
    env.current_subtasks = [env.subtask_dict[2]]
    env.rewarded_subtasks = [env.subtask_dict[2]]
    rewards: list[int] = []
    frames = [env.render()]

    for i in range(2):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(
        "tests/out/animations/labyrinth_compute_reward_intermediate_goal_agent_on_goal_2_right.gif",
        frames,
        loop=10,
    )

    assert rewards == [
        reward_config["movement_reward"] * 3
        + reward_config["agent_on_goal_reward"] * 3
        + reward_config["all_agents_on_goal_reward"],  # 1st step
        reward_config["movement_reward"] * 3,  # 2nd step
    ]


def test_compute_reward_intermediate_goal_agent_on_goal_2_stay():
    reward_config: RewardConfig = {
        "reward_option": "intermediate_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }

    init_pos: tuple[tuple[int, int], ...] = ((5, 3), (5, 4), (5, 5))
    p_intended_action: float = 1
    env = LabyrinthEnv(
        p_intended_action=p_intended_action,
        reward_config=reward_config,
        init_pos=init_pos,
    )

    actions = [
        [env.actions.right, env.actions.right, env.actions.right],
        [env.actions.stay, env.actions.stay, env.actions.stay],
    ]
    obs, _ = env.reset()
    env.current_subtasks = [env.subtask_dict[2]]
    env.rewarded_subtasks = [env.subtask_dict[2]]
    rewards: list[int] = []
    frames = [env.render()]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(
        "tests/out/animations/labyrinth_compute_reward_intermediate_goal_agent_on_goal_2_right.gif",
        frames,
        loop=10,
    )

    assert rewards == [
        reward_config["movement_reward"] * 3
        + reward_config["agent_on_goal_reward"] * 3
        + reward_config["all_agents_on_goal_reward"],  # 1st step
        0,  # 2nd step
    ]


def test_compute_reward_intermediate_goal_agent_on_goal_3():
    reward_config: RewardConfig = {
        "reward_option": "intermediate_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }

    init_pos: tuple[tuple[int, int], ...] = ((7, 3), (7, 4), (7, 5))
    p_intended_action: float = 1
    env = LabyrinthEnv(
        p_intended_action=p_intended_action,
        reward_config=reward_config,
        init_pos=init_pos,
    )

    action = [env.actions.right, env.actions.right, env.actions.right]

    obs, _ = env.reset()
    env.current_subtasks = [env.subtask_dict[3]]
    env.rewarded_subtasks = [env.subtask_dict[3]]
    rewards: list[int] = []
    frames = [env.render()]

    for i in range(1):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(
        "tests/out/animations/labyrinth_compute_reward_intermediate_goal_agent_on_goal_3.gif",
        frames,
        loop=10,
    )

    assert rewards == [
        reward_config["movement_reward"] * 3
        + reward_config["agent_on_goal_reward"] * 3
        + reward_config["all_agents_on_goal_reward"]
    ]
    assert terminated


def test_compute_reward_intermediate_goal_move_away() -> None:
    reward_config: RewardConfig = {
        "reward_option": "intermediate_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }

    init_pos: tuple[tuple[int, int], ...] = ((4, 1), (4, 2), (3, 3))
    p_intended_action: float = 1
    env = LabyrinthEnv(
        p_intended_action=p_intended_action,
        reward_config=reward_config,
        init_pos=init_pos,
    )

    action = [env.actions.left, env.actions.left, env.actions.stay]

    obs, _ = env.reset()
    env.current_subtasks = [env.subtask_dict[0]]
    env.rewarded_subtasks = [env.subtask_dict[0]]
    rewards: list[int] = []
    frames = [env.render()]

    for i in range(1):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(np.round(reward, 2))
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(
        "tests/out/animations/labyrinth_compute_reward_intermediate_goal_move_away.gif",
        frames,
        loop=10,
    )

    assert rewards == [
        reward_config["movement_reward"] * 2
        + reward_config["agent_move_away_from_goal_reward"] * 2
    ]


def test_compute_reward_final_goal_move_away() -> None:
    reward_config: RewardConfig = {
        "reward_option": "final_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }

    init_pos: tuple[tuple[int, int], ...] = ((4, 1), (4, 2), (3, 3))
    p_intended_action: float = 1
    env = LabyrinthEnv(
        p_intended_action=p_intended_action,
        reward_config=reward_config,
        init_pos=init_pos,
    )

    action = [env.actions.left, env.actions.left, env.actions.stay]

    obs, _ = env.reset()
    rewards: list[int] = []
    frames = [env.render()]

    for i in range(1):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(np.round(reward, 2))
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(
        "tests/out/animations/labyrinth_compute_reward_intermediate_goal_move_away.gif",
        frames,
        loop=10,
    )

    assert rewards == [reward_config["movement_reward"] * 2]


def test_compute_rewards_intermediate_goal_agent_on_goal_0() -> None:
    reward_config: RewardConfig = {
        "reward_option": "intermediate_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }

    init_pos: tuple[tuple[int, int], ...] = ((3, 1), (3, 2), (3, 3))
    p_intended_action: float = 1
    env = LabyrinthEnv(
        p_intended_action=p_intended_action,
        reward_config=reward_config,
        init_pos=init_pos,
    )

    action = [env.actions.right, env.actions.right, env.actions.stay]

    obs, _ = env.reset()
    env.current_subtasks = [env.subtask_dict[0]]
    env.rewarded_subtasks = [env.subtask_dict[0]]
    rewards: list[int] = []
    frames = [env.render()]

    for i in range(1):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(
        "tests/out/animations/labyrinth_compute_rewards_agent_on_goal_0.gif",
        frames,
        loop=10,
    )

    assert rewards == [
        reward_config["movement_reward"] * 2 + reward_config["agent_on_goal_reward"] * 2
    ]


def test_compute_rewards_final_goal_agent_on_goal_0() -> None:
    reward_config: RewardConfig = {
        "reward_option": "final_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }

    init_pos: tuple[tuple[int, int], ...] = ((3, 1), (3, 2), (3, 3))
    p_intended_action: float = 1
    env = LabyrinthEnv(
        p_intended_action=p_intended_action,
        reward_config=reward_config,
        init_pos=init_pos,
    )

    action = [env.actions.right, env.actions.right, env.actions.stay]

    obs, _ = env.reset()
    env.current_subtasks = [env.subtask_dict[0]]
    env.rewarded_subtasks = [env.subtask_dict[0]]
    rewards: list[int] = []
    frames = [env.render()]

    for i in range(1):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        frames.append(env.render())
        if terminated or truncated:
            break

    imageio.mimsave(
        "tests/out/animations/labyrinth_compute_rewards_agent_on_goal_0.gif",
        frames,
        loop=10,
    )

    assert rewards == [reward_config["movement_reward"] * 2]
