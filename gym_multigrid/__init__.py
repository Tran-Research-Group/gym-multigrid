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
            {"init_pos": (6, 6), "policy_type": "ego", "color": "green"},
            {"init_pos": (7, 7), "policy_type": "teammate", "color": "purple"},
            {"init_pos": (8, 8), "policy_type": "teammate", "color": "purple"},
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
    id="wildfire-v0",
    entry_point="gym_multigrid.envs:WildfireEnv",
)

# Labyrinth environment
# ----------------------------------------
register(
    id="multigrid-labyrinth-v0",
    entry_point="gym_multigrid.envs:LabyrinthEnv",
    kwargs={
        "num_agents": 3,
        "p_intended_action": 0.95,
        "init_pos": [(1, 3), (1, 4), (1, 5)],
        "subtask_config": [
            {
                "goal_group_index": 0,
                "next_subtask": 2,
                "assigned_agent_goal": {0: (4, 1), 1: (4, 2), 2: (4, 3)},
                "triggers": [
                    {
                        "condition": "agents_on_goals",
                        "action": "open",
                        "obj_type": "door",
                        "obj_group": 0,
                    }
                ],
            },
            {
                "goal_group_index": 1,
                "next_subtask": 2,
                "assigned_agent_goal": {0: (3, 5), 1: (3, 6), 2: (3, 7)},
                "triggers": [
                    {
                        "condition": "agents_on_goals",
                        "action": "open",
                        "obj_type": "door",
                        "obj_group": 1,
                    }
                ],
            },
            {
                "goal_group_index": 2,
                "next_subtask": 3,
                "assigned_agent_goal": {0: (6, 3), 1: (6, 4), 2: (6, 5)},
                "triggers": [
                    {
                        "condition": "agents_on_goals",
                        "action": "open",
                        "obj_type": "door",
                        "obj_group": 2,
                    }
                ],
            },
            {
                "goal_group_index": 3,
                "next_subtask": "terminal",
                "assigned_agent_goal": {0: (8, 3), 1: (8, 4), 2: (8, 5)},
                "triggers": [],
            },
        ],
        "obj_group_config": [
            {
                "obj_type": "goal",
                "group_index": 0,
                "pos": ((4, 1), (4, 2), (4, 3)),
                "color": "green",
                "fill_mode": None,
            },
            {
                "obj_type": "goal",
                "group_index": 1,
                "pos": ((3, 5), (3, 6), (3, 7)),
                "color": "green",
                "fill_mode": None,
            },
            {
                "obj_type": "goal",
                "group_index": 2,
                "pos": ((6, 3), (6, 4), (6, 5)),
                "color": "green",
                "fill_mode": None,
            },
            {
                "obj_type": "goal",
                "group_index": 3,
                "pos": ((8, 3), (8, 4), (8, 5)),
                "color": "green",
                "fill_mode": None,
            },
            {
                "obj_type": "door",
                "group_index": 0,
                "pos": ((5, 1), (5, 2), (5, 3)),
                "color": "light_grey",
                "fill_mode": None,
            },
            {
                "obj_type": "door",
                "group_index": 1,
                "pos": ((4, 5), (4, 6), (4, 7)),
                "color": "light_grey",
                "fill_mode": None,
            },
            {
                "obj_type": "door",
                "group_index": 2,
                "pos": ((7, 2), (7, 3), (7, 4), (7, 5), (7, 6)),
                "color": "light_grey",
                "fill_mode": None,
            },
            {
                "obj_type": "zone",
                "group_index": 0,
                "pos": ((2, 1), (2, 2), (2, 3)),
                "color": "blue",
                "fill_mode": None,
            },
            {
                "obj_type": "zone",
                "group_index": 1,
                "pos": ((2, 5), (2, 6), (2, 7)),
                "color": "red",
                "fill_mode": None,
            },
            {
                "obj_type": "wall",
                "group_index": 0,
                "pos": ((0, 0, 10, 9),),
                "color": "grey",
                "fill_mode": "empty",
            },
            {
                "obj_type": "wall",
                "group_index": 1,
                "pos": ((7, 1, 2, 1), (7, 7, 2, 1), (3, 4, 2, 1)),
                "color": "grey",
                "fill_mode": "filled",
            },
        ],
        "detector_config": [
            {
                "obj_type": "zone",
                "group_index": 0,
                "visual_detect_prob": 0.005,
                "radio_detect_prob": 0.0,
            },
            {
                "obj_type": "zone",
                "group_index": 1,
                "visual_detect_prob": 0.005,
                "radio_detect_prob": 0.06,
            },
        ],
        "reward_config": {
            "reward_option": "intermediate_goal",
            "movement_reward": -0.02,
            "agent_on_goal_reward": 0.2,
            "agent_move_away_from_goal_reward": -0.3,
            "all_agents_on_goal_reward": 1.0,
        },
        "observation_option": "intermediate_goal",
        "width": 10,
        "height": 9,
    },
)
