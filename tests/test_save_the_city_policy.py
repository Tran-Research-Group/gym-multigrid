import pytest
import numpy as np

from gym_multigrid.envs.save_the_city import SaveTheCityEnv, AgentConfig


def test_ego_only_agent():
    """Test environment with only ego agents (RL-controlled)."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        }
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Ego agent should not have a policy
    assert env.agents[0].policy is None
    assert env.agents[0].policy_class is None

    # Step should accept 1 action (for 1 ego agent)
    actions = [env.actions_set.NORTH]
    obs, reward, terminated, truncated, info = env.step(actions)

    assert obs is not None
    print("✓ Ego-only test passed")


def test_ego_with_random_actions():
    """Test ego agent with random actions (useful for debugging/testing)."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        }
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Run multiple steps with random actions
    for _ in range(10):
        # Generate random action for ego agent
        random_action = np.random.choice(list(env.actions_set))
        actions = [random_action]
        obs, reward, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            break

    print("✓ Ego with random actions test passed")


def test_teammate_only_agent():
    """Test environment with only teammate agents (policy-controlled)."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "teammate",
            "policy_name": "random",
        }
    ]

    # This should fail - need at least one ego agent
    with pytest.raises(ValueError, match="At least one ego agent is required"):
        env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)

    print("✓ Teammate-only validation test passed")


def test_mixed_ego_and_teammate():
    """Test environment with both ego and teammate agents."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "teammate",
            "policy_name": "random",
        },
        {
            "agent_type": "generalist",
            "policy_type": "teammate",
            "policy_name": "random",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Check ego agent (index determined by ego_agent_indices)
    ego_idx = env.ego_agent_indices[0]
    assert env.agents[ego_idx].policy is None
    assert env.agents[ego_idx].policy_class is None

    # Check teammate agents have policies
    for teammate_idx in env.non_ego_agent_indices:
        assert env.agents[teammate_idx].policy is not None
        assert env.agents[teammate_idx].policy.name == "random"

    # Step should accept 1 action (for 1 ego agent)
    # Teammates will generate their own actions
    actions = [env.actions_set.NORTH]
    obs, reward, terminated, truncated, info = env.step(actions)

    assert obs is not None
    print("✓ Mixed ego/teammate test passed")


def test_mixed_with_random_ego_actions():
    """Test mixed environment with random ego actions."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "teammate",
            "policy_name": "random",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Run episode with random ego actions
    for _ in range(20):
        # Random action for ego agent
        random_action = np.random.choice(list(env.actions_set))
        actions = [random_action]
        obs, reward, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            break

    print("✓ Mixed with random ego actions test passed")


def test_wrong_number_of_actions():
    """Test that step() validates the number of ego actions."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "teammate",
            "policy_name": "random",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Should fail - providing 2 actions but only 1 ego agent
    with pytest.raises(ValueError, match="Expected 1 ego actions, but received 2"):
        actions = [env.actions_set.NORTH, env.actions_set.SOUTH]
        env.step(actions)

    print("✓ Action count validation test passed")


def test_multiple_ego_agents():
    """Test environment with multiple ego agents."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "generalist",
            "policy_type": "teammate",
            "policy_name": "random",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Should have 2 ego agents
    assert len(env.ego_agent_indices) == 2
    assert len(env.non_ego_agent_indices) == 1

    # Step should accept 2 actions (for 2 ego agents)
    actions = [env.actions_set.NORTH, env.actions_set.SOUTH]
    obs, reward, terminated, truncated, info = env.step(actions)

    assert obs is not None
    print("✓ Multiple ego agents test passed")


def test_multiple_ego_with_random_actions():
    """Test multiple ego agents with random actions."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "generalist",
            "policy_type": "teammate",
            "policy_name": "random",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Run episode with random actions for both ego agents
    for _ in range(20):
        # Random actions for both ego agents
        random_actions = [
            np.random.choice(list(env.actions_set)),
            np.random.choice(list(env.actions_set)),
        ]
        obs, reward, terminated, truncated, info = env.step(random_actions)

        if terminated or truncated:
            break

    print("✓ Multiple ego with random actions test passed")


def test_all_random_policies():
    """Test all agents using random policy (1 ego, rest teammates with random)."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "teammate",
            "policy_name": "random",
        },
        {
            "agent_type": "generalist",
            "policy_type": "teammate",
            "policy_name": "random",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Run full episode with random ego actions + random teammate policies
    total_reward = 0
    for step in range(50):
        # Random action for ego agent
        random_action = np.random.choice(list(env.actions_set))
        actions = [random_action]
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}, total reward: {total_reward}")
            break

    print("✓ All random policies test passed")


def test_teammate_missing_policy_name():
    """Test that teammate agents require a policy_name."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "teammate",
            "policy_name": None,  # Invalid - teammate needs a policy
        },
    ]

    with pytest.raises(ValueError, match="must have a policy_name"):
        env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)

    print("✓ Teammate policy validation test passed")


def test_policy_reproducibility():
    """Test that policies produce reproducible results with same seed."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "teammate",
            "policy_name": "random",
        },
    ]

    # Run 1
    env1 = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs1, _ = env1.reset(seed=42)
    actions1 = [env1.actions_set.STILL]
    obs1_1, _, _, _, _ = env1.step(actions1)

    # Run 2 with same seed
    env2 = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs2, _ = env2.reset(seed=42)
    actions2 = [env2.actions_set.STILL]
    obs2_1, _, _, _, _ = env2.step(actions2)

    # Observations should be identical (teammate used same RNG)
    assert np.array_equal(obs1_1, obs2_1)

    print("✓ Reproducibility test passed")


# ============================================================================
# Tests for BuilderPolicy, FirefighterPolicy, GeneralistPolicy
# ============================================================================

def test_builder_policy_runs():
    """Test BuilderPolicy runs without errors."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "teammate",
            "policy_name": "builder",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Verify builder policy is assigned
    builder_idx = env.non_ego_agent_indices[0]
    assert env.agents[builder_idx].policy.name == "builder"

    # Run for multiple steps
    for _ in range(30):
        actions = [env.actions_set.STILL]
        obs, reward, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    print("✓ BuilderPolicy runs test passed")


def test_firefighter_policy_runs():
    """Test FirefighterPolicy runs without errors."""
    agent_configs = [
        {
            "agent_type": "builder",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "firefighter",
            "policy_type": "teammate",
            "policy_name": "firefighter",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Verify firefighter policy is assigned
    ff_idx = env.non_ego_agent_indices[0]
    assert env.agents[ff_idx].policy.name == "firefighter"

    # Run for multiple steps
    for _ in range(30):
        actions = [env.actions_set.STILL]
        obs, reward, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    print("✓ FirefighterPolicy runs test passed")


def test_generalist_policy_runs():
    """Test GeneralistPolicy runs without errors."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "generalist",
            "policy_type": "teammate",
            "policy_name": "generalist",
        },
    ]

    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Verify generalist policy is assigned
    gen_idx = env.non_ego_agent_indices[0]
    assert env.agents[gen_idx].policy.name == "generalist"

    # Run for multiple steps
    for _ in range(30):
        actions = [env.actions_set.STILL]
        obs, reward, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    print("✓ GeneralistPolicy runs test passed")


def test_all_intelligent_policies():
    """Test all three intelligent policies working together."""
    agent_configs = [
        {
            "agent_type": "firefighter",
            "policy_type": "ego",
            "policy_name": None,
        },
        {
            "agent_type": "builder",
            "policy_type": "teammate",
            "policy_name": "builder",
        },
        {
            "agent_type": "firefighter",
            "policy_type": "teammate",
            "policy_name": "firefighter",
        },
        {
            "agent_type": "generalist",
            "policy_type": "teammate",
            "policy_name": "generalist",
        },
    ]

    env = SaveTheCityEnv(size=12, num_buildings=5, agent_configs=agent_configs)
    obs, info = env.reset(seed=42)

    # Run full episode
    total_reward = 0
    for step in range(100):
        actions = [np.random.choice(list(env.actions_set))]
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}, total reward: {total_reward}")
            break

    print("✓ All intelligent policies test passed")


def test_deterministic_with_epsilon_zero():
    """Test policies are deterministic with epsilon=0."""
    from gym_multigrid.policy.save_the_city import BuilderPolicy
    from gym_multigrid.envs.save_the_city import SaveTheCityActions

    # Create two policies with same seed and epsilon=0
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    policy1 = BuilderPolicy(
        action_set=SaveTheCityActions,
        random_generator=rng1,
        epsilon=0.0,
    )
    policy2 = BuilderPolicy(
        action_set=SaveTheCityActions,
        random_generator=rng2,
        epsilon=0.0,
    )

    # Create environment to get observation
    agent_configs = [
        {"agent_type": "builder", "policy_type": "ego", "policy_name": None},
    ]
    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, _ = env.reset(seed=42)

    # Both policies should produce same action
    options = {'agent_pos': (5, 5)}
    action1 = policy1.act(obs, options)
    action2 = policy2.act(obs, options)

    assert action1 == action2, f"Actions differ: {action1} vs {action2}"
    print("✓ Deterministic with epsilon=0 test passed")


def test_policy_reset():
    """Test that policy reset clears internal state."""
    from gym_multigrid.policy.save_the_city import BuilderPolicy
    from gym_multigrid.envs.save_the_city import SaveTheCityActions

    policy = BuilderPolicy(
        action_set=SaveTheCityActions,
        random_generator=np.random.default_rng(42),
        epsilon=0.0,
    )

    # Set some internal state
    policy.current_target = (3, 4)
    policy.current_path = [(1, 1), (2, 2), (3, 3)]
    policy.path_index = 2

    # Reset
    policy.reset()

    # Verify state is cleared
    assert policy.current_target is None
    assert policy.current_path is None
    assert policy.path_index == 0

    print("✓ Policy reset test passed")


def test_builder_policy_with_epsilon():
    """Test BuilderPolicy randomness with different epsilon values."""
    from gym_multigrid.policy.save_the_city import BuilderPolicy
    from gym_multigrid.envs.save_the_city import SaveTheCityActions

    # Create environment to get observation
    agent_configs = [
        {"agent_type": "builder", "policy_type": "ego", "policy_name": None},
    ]
    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, _ = env.reset(seed=42)
    options = {'agent_pos': (5, 5)}

    # With epsilon=1.0, should always take random actions
    policy_random = BuilderPolicy(
        action_set=SaveTheCityActions,
        random_generator=np.random.default_rng(42),
        epsilon=1.0,
    )

    # Collect many actions - with epsilon=1.0 they should vary
    actions = [policy_random.act(obs, options) for _ in range(20)]
    unique_actions = set(actions)
    # Should have variety in actions (statistically very likely with 20 samples)
    assert len(unique_actions) > 1, "Expected variety in random actions"

    print("✓ BuilderPolicy epsilon test passed")


def test_utils_functions():
    """Test utility functions directly."""
    from gym_multigrid.policy.save_the_city import utils
    from gym_multigrid.core.world import SaveTheCityWorld

    world = SaveTheCityWorld  # Already an instance, not a class

    # Create environment to get observation
    agent_configs = [
        {"agent_type": "builder", "policy_type": "ego", "policy_name": None},
    ]
    env = SaveTheCityEnv(size=10, num_buildings=3, agent_configs=agent_configs)
    obs, _ = env.reset(seed=42)

    # Test extract_buildings_from_obs
    buildings = utils.extract_buildings_from_obs(obs, world)
    assert isinstance(buildings, list)
    # Should have found some buildings
    assert len(buildings) > 0

    # Test each building has required keys
    for b in buildings:
        assert 'pos' in b
        assert 'type' in b
        assert 'building_state' in b
        assert 'fire_rate' in b
        assert 'alive' in b

    # Test manhattan_distance
    assert utils.manhattan_distance((0, 0), (3, 4)) == 7
    assert utils.manhattan_distance((5, 5), (5, 5)) == 0

    # Test is_adjacent_to_target
    assert utils.is_adjacent_to_target((1, 1), (1, 2)) == True
    assert utils.is_adjacent_to_target((1, 1), (2, 2)) == False

    # Test get_adjacent_positions
    adj = utils.get_adjacent_positions((5, 5))
    assert len(adj) == 4
    assert (5, 4) in adj  # North
    assert (6, 5) in adj  # East
    assert (5, 6) in adj  # South
    assert (4, 5) in adj  # West

    print("✓ Utils functions test passed")


if __name__ == "__main__":
    test_ego_only_agent()
    test_ego_with_random_actions()
    test_teammate_only_agent()
    test_mixed_ego_and_teammate()
    test_mixed_with_random_ego_actions()
    test_wrong_number_of_actions()
    test_multiple_ego_agents()
    test_multiple_ego_with_random_actions()
    test_all_random_policies()
    test_teammate_missing_policy_name()
    test_policy_reproducibility()
    # New policy tests
    test_builder_policy_runs()
    test_firefighter_policy_runs()
    test_generalist_policy_runs()
    test_all_intelligent_policies()
    test_deterministic_with_epsilon_zero()
    test_policy_reset()
    test_builder_policy_with_epsilon()
    test_utils_functions()
    print("\n✅ All tests passed!")
