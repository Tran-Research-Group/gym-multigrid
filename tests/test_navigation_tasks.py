from pathlib import Path

import numpy as np
import pytest
import yaml
from gym_multigrid.envs.mdp import ProjectMDP
from gym_multigrid.envs.team_navigation import TeamNavigationEnv
from gym_multigrid.utils.subtasks import NavigationTaskCatalog


def _task(task_id, from_state, to_state, goals, init_positions=None):
    return {
        "id": task_id,
        "from_state": from_state,
        "to_state": to_state,
        "goal_positions": goals,
        "init_state_dist": {
            "states": [
                [
                    *(init_positions or [(1, 1), (2, 1), (3, 1)]),
                ]
            ],
            "probs": [1.0],
        },
    }


def test_delay_agents_are_uniformly_sampled_for_episode():
    env = TeamNavigationEnv(
        width=8,
        height=8,
        n_agents=4,
        n_agents_delay=2,
        agent_delay_prob=1.0,
    )

    try:
        selected_sets = set()
        for seed in range(20):
            env.reset(seed=seed)
            selected = tuple(np.flatnonzero(env._delay_candidate_agents))
            assert len(selected) == 2
            assert tuple(np.flatnonzero(env._delayed_agents)) == selected
            selected_sets.add(selected)
            env.step(np.full(4, int(env.actions.STAY), dtype=int))
            assert tuple(np.flatnonzero(env._delay_candidate_agents)) == selected

        assert len(selected_sets) > 1
        assert {agent for selected in selected_sets for agent in selected} == set(
            range(4)
        )
    finally:
        env.close()


@pytest.mark.parametrize("n_agents_delay", [-1, 5])
def test_delay_agent_count_must_fit_team_size(n_agents_delay):
    with pytest.raises(ValueError, match="n_agents_delay"):
        TeamNavigationEnv(
            width=8,
            height=8,
            n_agents=4,
            n_agents_delay=n_agents_delay,
        )


def test_navigation_catalog_distinguishes_branching_transitions():
    catalog = NavigationTaskCatalog.from_configs(
        [
            _task("0_to_1", 0, 1, [(1, 2), (2, 2), (3, 2)]),
            _task(
                "1_to_2",
                1,
                2,
                [(1, 3), (2, 3), (3, 3)],
                init_positions=[(1, 2), (2, 2), (3, 2)],
            ),
        ],
        num_agents=3,
        width=5,
        height=5,
    )

    assert catalog.task_for(0, 1).task_id == "0_to_1"
    assert catalog.task_for(1, 2).task_id == "1_to_2"


def test_navigation_catalog_rejects_disconnected_successor_spawn():
    with pytest.raises(ValueError, match="must start from the predecessor"):
        NavigationTaskCatalog.from_configs(
            [
                _task("0_to_1", 0, 1, [(1, 2), (2, 2), (3, 2)]),
                _task("1_to_2", 1, 2, [(1, 4), (2, 4), (3, 4)]),
            ],
            num_agents=3,
            width=5,
            height=5,
        )


def test_learned_hallway_catalog_only_configures_root_spawn():
    config_path = (
        Path(__file__).parents[1]
        / "gym_multigrid"
        / "envs"
        / "maps"
        / "team_navigation"
        / "navigation_hallway_learned.yaml"
    )
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    catalog = NavigationTaskCatalog.from_configs(
        config,
        num_agents=3,
        width=8,
        height=7,
    )

    assert catalog.task_for(0, 1).init_state_dist is not None
    assert all(
        catalog.task_for(from_state, to_state).init_state_dist is None
        for from_state, to_state in [(1, 2), (2, 3), (3, 4)]
    )


def test_successor_reset_uses_learned_spawn_distribution():
    catalog = NavigationTaskCatalog.from_configs(
        [
            {
                "id": "0_to_1",
                "from_state": 0,
                "to_state": 1,
                "goal_positions": [[4, 1], [4, 3], [4, 5]],
                "init_state_dist": {
                    "states": [[[1, 1], [1, 3], [1, 5]]],
                    "probs": [1.0],
                },
            },
            {
                "id": "1_to_2",
                "from_state": 1,
                "to_state": 2,
                "goal_positions": [[5, 1], [5, 3], [5, 5]],
            },
        ],
        num_agents=3,
        width=8,
        height=7,
    )
    env = TeamNavigationEnv(
        width=8,
        height=7,
        n_agents=3,
        navigation_task_catalog=catalog,
        hallway=True,
    )
    assert env.observation_space.shape == (3, 5)
    learned_spawn_dist = {
        "states": [[[4, 1], [4, 3], [4, 5]]],
        "probs": [1.0],
    }

    try:
        _, info = env.reset(
            options={
                "navigation_task_transition": [1, 2],
                "navigation_init_state_dist": learned_spawn_dist,
            }
        )
        assert info["final_state"] == (
            (4, 1),
            (4, 3),
            (4, 5),
        )

        with pytest.raises(ValueError, match="has no initial state distribution"):
            env.reset(options={"navigation_task_transition": [1, 2]})
    finally:
        env.close()


def test_one_to_one_goal_assignment_accepts_any_agent_permutation():
    goal_positions = [(2, 2), (3, 2), (4, 2)]
    catalog = NavigationTaskCatalog.from_configs(
        [
            {
                "from_state": 0,
                "to_state": 1,
                "goal_positions": goal_positions,
                "goal_assignment": "one_to_one",
                "init_state_dist": {
                    "states": [[[3, 3], [2, 3], [4, 3]]],
                    "probs": [1.0],
                },
            },
            {
                "from_state": 1,
                "to_state": 2,
                "goal_positions": [(2, 3), (3, 3), (4, 3)],
                "init_state_dist": {
                    "states": [[[3, 2], [2, 2], [4, 2]]],
                    "probs": [1.0],
                },
            },
        ],
        num_agents=3,
        width=9,
        height=9,
    )
    env = TeamNavigationEnv(
        map_name="3ga_1r_small",
        n_agents=3,
        navigation_task_catalog=catalog,
    )

    try:
        obs, _ = env.reset(options={"navigation_task_transition": [0, 1]})
        assert env.observation_space.shape == (3, 9)
        assert obs.shape == (3, 9)

        _, _, terminated, _, info = env.step(np.full(3, int(env.actions.UP), dtype=int))

        assert terminated
        assert info["task_completed"]
        assert set(info["final_state"]) == set(goal_positions)
        assert len(set(info["final_state"])) == 3
        assert info["final_state"] == ((3, 2), (2, 2), (4, 2))
    finally:
        env.close()


def test_one_to_one_successor_spawn_must_be_a_goal_permutation():
    with pytest.raises(ValueError, match="one-to-one assignments"):
        NavigationTaskCatalog.from_configs(
            [
                {
                    "from_state": 0,
                    "to_state": 1,
                    "goal_positions": [(2, 2), (3, 2), (4, 2)],
                    "goal_assignment": "one_to_one",
                },
                {
                    "from_state": 1,
                    "to_state": 2,
                    "goal_positions": [(2, 3), (3, 3), (4, 3)],
                    "init_state_dist": {
                        "states": [[[2, 2], [2, 2], [4, 2]]],
                        "probs": [1.0],
                    },
                },
            ],
            num_agents=3,
            width=9,
            height=9,
        )


def test_project_mdp_supports_branching_and_merging_transitions():
    mdp = ProjectMDP(
        msg_budget_per_agent=[0, 1],
        states=[0, 1, 2, 3],
        transitions=[(0, 1), (0, 2), (1, 3), (2, 3)],
        initial_state=0,
        goal_states=[3],
    )

    assert (0, (1, 0)) in mdp.successor_map
    assert (0, (2, 1)) in mdp.successor_map
    assert mdp.goal_states == {3}
    assert np.array_equal(mdp.reset()[0], np.array([0, False]))
