# Labyrinth Environment
Multi-agent labyrinth env with multiple goals and zones.

### Example
``` python
import gymnasium as gym

env = gym.make("multigrid-labyrinth-v0")
```

## Observation
The format of the observation is a dictionary of each agent's observation with the agent's index as the key.
- The observation is the positions of the final goal and agents if the observation option is "final_goal".
- The observation is the positions of all the goals and agents if the observation option is "intermediate_goal".

### Example
``` python
# Observation option is "final_goal"
observation_space = dict({
    "0": Box(low=np.zeros(4), high=np.array([9, 8, 9, 8]), dtype=np.int_),
    "1": Box(low=np.zeros(4), high=np.array([9, 8, 9, 8]), dtype=np.int_),
    "2": Box(low=np.zeros(4), high=np.array([9, 8, 9, 8]), dtype=np.int_),
})

# Observation option is "intermediate_goal"
observation_space = dict({
    "0": Box(low=np.zeros(10), high=np.array([9, 8] * 5), dtype=np.int_),
    "1": Box(low=np.zeros(10), high=np.array([9, 8] * 5), dtype=np.int_),
    "2": Box(low=np.zeros(10), high=np.array([9, 8] * 5), dtype=np.int_),
})
```

## Actions
- There are five actions: "stay", "up", "right", "down", and "left".
- The action space is MultiDiscrete([5, 5, 5]) for three agents.

## Reward
- Movement penalty for each agent if an action is not "stay".
- Reward for each agent if it is on its assigned goal.
- Penalty for each agent if it moves away from its assigned goal though it was on it.
- Reward for all agents if they are on their assigned goals on the same goal group and unlock the door.

These rewards are specified in the `reward_config` parameter.

### Example
``` python
reward_config: RewardConfig = {
    "reward_option": "final_goal",
    "movement_reward": -0.02,
    "agent_on_goal_reward": 0.2,
    "agent_move_away_from_goal_reward": -0.3,
    "all_agents_on_goal_reward": 1.0,
}
```
## Subtasks
- The subtasks are defined in the `subtask_config` parameter.
- Each subtask has the following
    - `next_subtask`: Next subtask index or "terminal". If a subtask is a terminal subtask, the episode ends.
    - `goal_group_index`: Goal group index.
    - `assigned_agent_goal`: Assigned agent goal positions as a dictionary with the agent index as the key and the assigned goal position as the value.
    - `triggers`: Triggers to open the door.

### Trigger
- The trigger has the following
    - `condition`: Condition to trigger the action.
    - `action`: Action to call for the object.
    - `obj_type`: Type of the object to call the action.
    - `obj_group`: Group index of the object to call the action.

### Example
``` python
subtask_config: list[SubtaskConfig] = [
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
]
```

## Object Groups
- The object groups are defined in the `obj_group_config` parameter.
- Each object group has the following
    - `obj_type`: Type of the object.
    - `group_index`: Group index of the object.
    - `pos`: Positions of the objects.
    - `color`: Color of the objects.
    - `fill_mode`: Fill mode of the object.

### Example
``` python
obj_group_config: list[ObjectGroupConfig] = [
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
]
```

## Detectors
- The detectors are defined in the `detector_config` parameter.
- Each detector has the following
    - `obj_type`: Type of the object where the detector is placed.
    - `group_index`: Group index of the object where the detector is placed.
    - `visual_detect_prob`: Probability of the visual detection.
    - `radio_detect_prob`: Probability of the radio detection.

### Example
``` python
detector_config: list[DetectorConfig] = [
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
]
```