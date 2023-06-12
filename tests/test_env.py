from gym_multigrid.envs.collect_game import CollectGame3Obj2Agent


def test_collect_game() -> None:
    """Test collect_game()"""
    env = CollectGame3Obj2Agent()

    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            break
