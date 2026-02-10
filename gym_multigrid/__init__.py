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

# Wildfire environment
# ----------------------------------------
register(
    id="multigrid-wildfire-v0",
    entry_point="gym_multigrid.envs:WildfireEnv",
)

# Labyrinth environment
# ----------------------------------------
register(
    id="multigrid-independent-subtask-labyrinth-v0",
    entry_point="gym_multigrid.envs:LabyrinthEnv",
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

# Rooms environment
# ----------------------------------------
register(
    id="multigrid-rooms-v0",
    entry_point="gym_multigrid.envs:RoomsEnv",
    max_episode_steps=100,
    kwargs={
        "spawn_type": 0,
        "layout_config": {
            "field_map": [
                "#############",
                "#     #     #",
                "#     #     #",
                "#           #",
                "#     #     #",
                "#     #     #",
                "## ####     #",
                "#     ### ###",
                "#     #     #",
                "#     #     #",
                "#           #",
                "#     #     #",
                "#############",
            ],
            "spawn_configs": [
                {
                    "agent": (9, 3),
                    "goal": {"pos": (3, 9), "reward": 1.0, "absorbing": True},
                },
                {
                    "agent": (11, 1),
                    "goal": {"pos": (7, 9), "reward": 1.0, "absorbing": True},
                },
                {
                    "agent": (9, 3),
                    "goal": {"pos": (9, 9), "reward": 1.0, "absorbing": True},
                },
                {
                    "agent": (3, 9),
                    "goal": {"pos": (9, 4), "reward": 1.0, "absorbing": True},
                    "lavas": [
                        {"pos": (8, 4), "reward": 0, "absorbing": False},
                        {"pos": (9, 2), "reward": 0, "absorbing": False},
                        {"pos": (11, 1), "reward": 0, "absorbing": False},
                        {"pos": (5, 3), "reward": 0, "absorbing": False},
                        {"pos": (3, 5), "reward": 0, "absorbing": False},
                        {"pos": (3, 2), "reward": 0, "absorbing": False},
                        {"pos": (5, 9), "reward": -1, "absorbing": True},
                        {"pos": (3, 8), "reward": -1, "absorbing": True},
                        {"pos": (2, 11), "reward": -1, "absorbing": True},
                        {"pos": (10, 8), "reward": -1, "absorbing": True},
                        {"pos": (8, 9), "reward": -1, "absorbing": True},
                        {"pos": (7, 11), "reward": -1, "absorbing": True},
                    ],
                    "holes": [
                        {"pos": (7, 3), "reward": 0, "absorbing": False},
                        {"pos": (10, 5), "reward": 0, "absorbing": False},
                        {"pos": (8, 6), "reward": 0, "absorbing": False},
                        {"pos": (4, 4), "reward": -1, "absorbing": True},
                        {"pos": (2, 3), "reward": -1, "absorbing": True},
                        {"pos": (1, 1), "reward": -1, "absorbing": True},
                        {"pos": (2, 7), "reward": 0, "absorbing": False},
                        {"pos": (1, 9), "reward": 0, "absorbing": False},
                        {"pos": (4, 10), "reward": 0, "absorbing": False},
                        {"pos": (7, 8), "reward": -1, "absorbing": True},
                        {"pos": (9, 10), "reward": -1, "absorbing": True},
                        {"pos": (11, 11), "reward": -1, "absorbing": True},
                    ],
                },
            ],
        },
        "state_representation": "vectorized_tensor",
        "reward_config": {
            "step_penalty": 0.01,
            "sum_reward": True,
        },
        "tile_size": 32,
        "render_mode": "rgb_array",
    },
)

# LBF environment
# ----------------------------------------
register(
    id="multigrid-lbf-v0",
    entry_point="gym_multigrid.envs:LBFGameEnv",
    max_episode_steps=400,
    kwargs={
        "agent_config": {
            "level": [1, 1, 2],
            "init_pos": [(1, 1), (1, 2), (1, 3)],
            "index": [4, 5, 6],
        },
        "fruit_config": {
            "num_fruit": [8, 8, 8],
            "fruits_index": [0, 1, 2],
            "fruits_reward": [1, 2, 3]
        },
        "layout_config": {
            "size": 19,
            "num_rooms": 4,
            "field_map": [
                "###################",
                "#     1  # 2      #",
                "#  1     #  1     #",
                "#       G#     1  #",
                "#  2    G#   1    #",
                "#       G#        #",
                "#    1   #  2   2 #",
                "# 1      #        #",
                "#     1  #  GGG   #",
                "###################",
                "#  GGG   #   2    #",
                "#      3 #        #",
                "# 3      #G    2  #",
                "#    3   #G  3    #",
                "#        #G       #",
                "# 3      #   3    #",
                "#   3    #  2  3  #",
                "#     2  #        #",
                "###################",
            ],
            "room_configs": [
                {
                    "flag_positions": [(8, 3), (8, 4), (8, 5)],
                    "fruit_count": [5, 1, 0],
                    "wall_positions": [(9, 3), (9, 4), (9, 5)],
                },
                {
                    "flag_positions": [(12, 8), (13, 8), (14, 8)],
                    "fruit_count": [3, 3, 0],
                    "wall_positions": [(12, 9), (13, 9), (14, 9)],
                },
                {
                    "flag_positions": [(10, 12), (10, 13), (10, 14)],
                    "fruit_count": [0, 3, 3],
                    "wall_positions": [(9, 12), (9, 13), (9, 14)],
                },
                {
                    "flag_positions": [(3, 10), (4, 10), (5, 10)],
                    "fruit_count": [0, 1, 5],
                    "wall_positions": [(3, 9), (4, 9), (5, 9)],
                }
            ]
        },
        "reward_config": {
            "step_penalty": 0.01,
            "sum_reward": True,
        },
    },
)
