from gym_multigrid.envs.collect_game import CollectGameEnv


def test_collect_game() -> None:
    """Test collect_game()"""
    env = CollectGameEnv()

    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
