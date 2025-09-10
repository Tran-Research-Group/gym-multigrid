from gymnasium.envs.registration import register

# Collect game with 2 agents and 3 object types
# ----------------------------------------
register(
    id="multigrid-collect-v0",
    entry_point="gym_multigrid.envs:CollectGameEvenDist",
    max_episode_steps=100,
    kwargs={
        "size": 10,
        "num_balls": 15,
        "agents_index": [3, 5],  # green, purple
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": False,
    },
)

# Collect game with single agent and 3 object types
# ----------------------------------------
register(
    id="multigrid-collect-single-v0",
    entry_point="gym_multigrid.envs:CollectGameEvenDist",
    max_episode_steps=100,
    kwargs={
        "size": 10,
        "num_balls": 15,
        "agents_index": [3],  # green
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": False,
    },
)

# Collect game with 2 agents and 3 object types clustered in different quadrants of the grid
# ----------------------------------------
register(
    id="multigrid-collect-quadrants-v0",
    entry_point="gym_multigrid.envs:CollectGameQuadrants",
    max_episode_steps=100,
    kwargs={
        "size": 10,
        "num_balls": 15,
        "agents_index": [3, 5],  # green, purple
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": False,
    },
)

# Collect game with 2 agents and 3 object types clustered in four rooms
# ----------------------------------------
register(
    id="multigrid-collect-rooms-v0",
    entry_point="gym_multigrid.envs:CollectGameRooms",
    max_episode_steps=100,
    kwargs={
        "size": 10,
        "num_balls": 15,
        "agents_index": [3, 5],  # green, purple
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": False,
    },
)

# Collect game with 2 agents and 3 object types clustered differently in four rooms
# Episode has a fixed horizon instead of terminating after collecting all objects
# ----------------------------------------
register(
    id="multigrid-collect-rooms-fixed-horizon-v0",
    entry_point="gym_multigrid.envs:CollectGameRoomsFixedHorizon",
    max_episode_steps=100,
    kwargs={
        "size": 10,
        "num_balls": 15,
        "agents_index": [3, 5],  # green, purple
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": False,
    },
)

# Collect game with 2 agents and 3 object types clustered differently in four rooms
# Episode has a fixed horizon and objects respawn after collection
# ----------------------------------------
register(
    id="multigrid-collect-rooms-respawn-v0",
    entry_point="gym_multigrid.envs:CollectGameRoomsFixedHorizon",
    max_episode_steps=50,
    kwargs={
        "size": 10,
        "num_balls": 15,
        "agents_index": [3, 5],  # green, purple
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": True,
    },
)

# Collect game with 2 agents and 3 object types
# Episode has a fixed horizon and objects respawn after collection
# ----------------------------------------
register(
    id="multigrid-collect-respawn-v0",
    entry_point="gym_multigrid.envs:CollectGameEvenDist",
    max_episode_steps=50,
    kwargs={
        "size": 10,
        "num_balls": 15,
        "agents_index": [3, 5],  # green, purple
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": True,
    },
)

# Collect game with 2 agents and 3 object types clustered in different quadrants of the grid
# Episode has a fixed horizon and objects respawn after collection
# ----------------------------------------
register(
    id="multigrid-collect-respawn-clustered-v0",
    entry_point="gym_multigrid.envs:CollectGameQuadrantsRespawn",
    max_episode_steps=50,
    kwargs={
        "size": 10,
        "num_balls": 15,
        "agents_index": [3, 5],  # green, purple
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": True,
    },
)

register(
    id="multigrid-collect-quadrants15-v0",
    entry_point="gym_multigrid.envs:CollectGameQuadrants",
    kwargs={
        "size": 15,
        "num_balls": 30,
        "agents_index": [3, 5],  # green, purple
        "balls_index": [0, 1, 2],  # red, orange, yellow
        "balls_reward": [1, 1, 1],
        "respawn": False,
    },
)

# PreyPred environment
# ----------------------------------------
register(
    id="multigrid-preypred-v0",
    entry_point="gym_multigrid.envs:PreyPredEnv",
    kwargs={
        "max_episode_steps": 300,
        "observation_config": {"encode_prey_areas": True},
        "pred_configs": [
            {
                "init_pos": (6, 6),
                "policy_type": "ego",
                "target_preys": [1, 1, 1, 1],
                "color": "green",
            },
            {
                "init_pos": (7, 7),
                "policy_type": "teammate",
                "target_preys": [1, 1, 1, 0],
                "color": "purple",
            },
            {
                "init_pos": (8, 8),
                "policy_type": "teammate",
                "target_preys": [0, 0, 1, 1],
                "color": "purple",
            },
        ],
        "prey_configs": [
            {
                "type": "easy_prey",
                "territory_dims": (4, 4),
                "territory_left_top_corner": (2, 2),
                "color": "yellow",
            },
            {
                "type": "easy_prey",
                "territory_dims": (4, 4),
                "territory_left_top_corner": (9, 2),
                "color": "yellow",
            },
            {
                "type": "hard_prey",
                "territory_dims": (4, 4),
                "territory_left_top_corner": (2, 9),
                "color": "red",
            },
            {
                "type": "hard_prey",
                "territory_dims": (4, 4),
                "territory_left_top_corner": (9, 9),
                "color": "red",
            },
        ],
        "prey_types": [
            {
                "name": "easy_prey",
                "policy": "random",
                "capture_reward": 1.0,
                "num_required_preds_capture": 1,
                "num_required_preds_fix": 1,
            },
            {
                "name": "hard_prey",
                "policy": "random",
                "capture_reward": 1.0,
                "num_required_preds_capture": 2,
                "num_required_preds_fix": 1,
            },
        ],
        "render_mode": "rgb_array",
    },
)


# Maze environment
# ----------------------------------------
register(
    id="multigrid-maze-v0",
    entry_point="gym_multigrid.envs:MazeEnv",
    max_episode_steps=100,
    kwargs={
        "num_agents": 1,
        "layout_config": {
            "width": 10,
            "height": 10,
            "flag_positions": [(9, 9)],
            "init_agent_positions": [(5, 5)],
            "wall_positions": [],
        },
        "reward_config": {
            "flag_reward": 1.0,
            "wall_penalty_ratio": 0.0,
            "step_penalty_ratio": 0.01,
        },
        "observation_mode": "tensor",
        "render_mode": "rgb_array",
    },
)

# Wildfire environment
# ----------------------------------------
register(
    id="multigrid-wildfire-v0",
    entry_point="gym_multigrid.envs:WildfireEnv",
)

# Team Navigation environment
# ----------------------------------------
register(
    id="multigrid-team-navigation-5-task-v0",
    entry_point="gym_multigrid.envs:TeamNavigationEnv",
)

# One-step coordination environment (with groups)
# ----------------------------------------
register(
    id="multigrid-one-step-coordination-v0",
    entry_point="gym_multigrid.envs:OneStepCoordinationEnv",
)

# One-step coordination environment (no groups)
# ----------------------------------------
register(
    id="multigrid-one-step-coordination-no-groups-v0",
    entry_point="gym_multigrid.envs:OneStepCoordinationNoGroupsEnv",
)


# TWo agent dependent subtask test env
# ----------------------------------------
register(
    id="two_agent_two_task_small-v0",
    entry_point="gym_multigrid.envs:TwoAgentTwoTaskSmallEnv",
)
