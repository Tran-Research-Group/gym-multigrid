from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar, TypedDict, Optional

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

from gym_multigrid.core.agent import NavigationActions, ActionsT, Agent, NAV_DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import AgentGoal, Wall, WorldObjT, Zone
from gym_multigrid.core.object import SimpleDoor as Door
from gym_multigrid.core.world import WorldT, LabyrinthWorld
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing import Position


class TriggerConfig(TypedDict):
    condition: str
    action: str
    obj_type: str
    obj_group: int


class SubtaskConfig(TypedDict):
    next_subtask: int | Literal["terminal"]
    goal_group_index: int
    assigned_agent_goal: dict[int, tuple[int, int]]
    triggers: list[TriggerConfig]


class ObjectGroupConfig(TypedDict):
    obj_type: str
    group_index: int
    pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...]
    color: str
    fill_mode: Literal["empty", "filled"] | None


class DetectorConfig(TypedDict):
    obj_type: str
    group_index: int
    visual_detect_prob: float
    radio_detect_prob: float


class RewardConfig(TypedDict):
    reward_option: Literal["final_goal", "intermediate_goal"]
    movement_reward: float
    agent_on_goal_reward: float
    agent_move_away_from_goal_reward: float
    all_agents_on_goal_reward: float


class ObjectGroup(ABC):
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...],
        color: str,
        fill_mode: Literal["empty", "filled"] = "filled",
        object_options: dict[str, WorldObjT] = {
            "goal": AgentGoal,
            "door": Door,
            "zone": Zone,
            "wall": Wall,
        },
    ) -> None:
        """
        Initializes the object group.

        Parameters
        ----------
        obj_type : str
            Type of the object.
        group_index : int
            Group index of the object.
        pos : tuple[tuple[int, int] | tuple[int, int, int, int], ...]
            Positions of the objects.
            A tuple element can be either a tuple of two integers or a tuple of four integers.
            - (x, y): Position of the object.
            - (x, y, w, h): Position and size of the object.
        fill_mode : Literal["empty", "filled"] = "filled"
            Fill mode of the object.
            - "empty": Empty fill mode.
            - "filled": Filled fill mode.
        """
        self.obj_type: str = obj_type
        self.group_index: int = group_index
        self.color: str = color
        self.object_options: dict[str, WorldObjT] = object_options

        pos_list: list[tuple[int, int]] = []
        for p in pos:
            if len(p) == 2:
                pos_list.append(p)
            elif len(p) == 4:
                if fill_mode == "empty":
                    pos_list += self._rect_empty(*p)
                elif fill_mode == "filled":
                    pos_list += self._rect_filled(*p)
                else:
                    raise ValueError(f"Invalid fill mode: {fill_mode}")
            else:
                raise ValueError(f"Invalid position: {p}. The length should be 2 or 4.")

        self.pos: tuple[tuple[int, int], ...] = tuple(pos_list)

    def put_objects(self, grid: Grid, world: WorldT) -> None:
        """
        Places the objects on the grid.

        Parameters
        ----------
        grid : Grid
            Global grid from the env to place the objects.
        world : WorldT
            World to place the objects.
        """
        for pos in self.pos:
            obj: WorldObjT = self._init_obj(world)
            self._put_obj(grid, pos, obj)

    def _init_obj(self, world: WorldT) -> WorldObjT:
        """
        Defines the initialization of the object.
        """
        return self.object_options[self.obj_type](
            world, type=self.obj_type, color=self.color
        )

    def _put_obj(self, grid: Grid, pos: tuple[int, int], obj: WorldObjT) -> None:
        """
        Places the object on the grid.

        Parameters
        ----------
        grid : Grid
            Global grid from the env to place the object.
        pos : tuple[int, int]
            Position to place the object.
        obj : WorldObjT
            Object to place.
        """

        obj.init_pos = pos
        obj.pos = pos
        grid.set(*pos, obj)

    def apply_obj_action(
        self, action: str, grid: Grid, args: dict[str, Any] = {}
    ) -> Any:
        """
        Applies the action to each object in the group.

        Parameters
        ----------
        action : str
            Action to apply.
        args : dict[str, Any] = {}
            Arguments for the action.

        Returns
        -------
        outputs : list[Any]
            Outputs of the action for each object in the group.
        """
        outputs: list[Any] = []
        for pos in self.pos:
            obj: WorldObjT = grid.get(*pos)
            outputs.append(getattr(obj, action)(**args))

        return outputs

    def call_action(self, action: str, args: dict[str, Any] = {}) -> Any:
        return getattr(self, action)(**args)

    def _horz_fill(
        self,
        x: int,
        y: int,
        length: int,
    ) -> list[tuple[int, int]]:
        pos_list: list[tuple[int, int]] = [(x + i, y) for i in range(length)]
        return pos_list

    def _vert_fill(
        self,
        x: int,
        y: int,
        length: int,
    ) -> list[tuple[int, int]]:
        pos_list: list[tuple[int, int]] = [(x, y + i) for i in range(length)]
        return pos_list

    def _rect_empty(self, x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
        pos_list: list[tuple[int, int]] = (
            self._horz_fill(x, y, w)
            + self._horz_fill(x, y + h - 1, w)
            + self._vert_fill(x, y, h)
            + self._vert_fill(x + w - 1, y, h)
        )

        return pos_list

    def _rect_filled(self, x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
        pos_list: list[tuple[int, int]] = [
            (x + i, y + j) for i in range(w) for j in range(h)
        ]

        return pos_list


ObjGroupT = TypeVar("ObjGroupT", bound=ObjectGroup)


class Subtask:
    def __init__(
        self,
        next_subtask: int | Literal["terminal"],
        goal_group_index: int,
        assigned_agent_goal: dict[int, tuple[int, int]],
        triggers: list[TriggerConfig],
    ) -> None:
        self.next_subtask: int | Literal["terminal"] = next_subtask
        self.goal_group_index: int = goal_group_index
        self.assigned_agent_goal: dict[int, tuple[int, int]] = assigned_agent_goal
        self.triggers: list[Trigger] = [Trigger(**trigger) for trigger in triggers]


class Trigger:
    def __init__(
        self,
        condition: str,
        action: str,
        obj_type: str,
        obj_group: int,
    ) -> None:
        self.condition: str = condition
        self.action: str = action
        self.obj_type: str = obj_type
        self.obj_group: int = obj_group

        self.triggered: bool = False

    def trigger_action(
        self, obj_group_dict: dict[str, dict[int, ObjGroupT]], grid: Grid
    ) -> None:
        obj_group_dict[self.obj_type][self.obj_group].apply_obj_action(
            self.action, grid
        )

        self.triggered = True

    def is_condition_satisfied(self, agents: list[Agent], subtask: Subtask) -> bool:
        return getattr(self, self.condition)(agents, subtask)

    def agents_on_goals(self, agents: list[Agent], subtask: Subtask) -> bool:
        for agent_index, goal_pos in subtask.assigned_agent_goal.items():
            if (
                agents[agent_index].pos[0] != goal_pos[0]
                or agents[agent_index].pos[1] != goal_pos[1]
            ):
                return False
            else:
                pass

        return True


class Detector:
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        visual_detect_prob: float,
        radio_detect_prob: float,
    ) -> None:
        self.obj_type: str = obj_type
        self.group_index: int = group_index
        self.visual_detect_prob: float = visual_detect_prob
        self.radio_detect_prob: float = radio_detect_prob

    def detect_agents(
        self,
        agents: list[Agent],
        obj_group_dict: dict[str, dict[int, ObjGroupT]],
        random_generator: np.random.Generator,
    ) -> bool:
        obj_group: ObjGroupT = obj_group_dict[self.obj_type][self.group_index]
        for agent in agents:
            if self.detect_agent(agent, obj_group, random_generator):
                return True
            else:
                pass

        return False

    def detect_agent(
        self,
        agent: list[Agent],
        obj_group: ObjGroupT,
        random_generator: np.random.Generator,
    ) -> bool:
        if (agent.pos[0], agent.pos[1]) in obj_group.pos:
            visual_detect: bool = random_generator.uniform() < self.visual_detect_prob
            radio_detect: bool = random_generator.uniform() < self.radio_detect_prob
            return visual_detect or radio_detect
        else:
            return False


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

reward_config: RewardConfig = {
    "reward_option": "intermediate_goal",
    "movement_reward": -0.02,
    "agent_on_goal_reward": 0.2,
    "agent_move_away_from_goal_reward": -0.3,
    "all_agents_on_goal_reward": 1.0,
}


class LabyrinthEnv(MultiGridEnv):
    """
    # Labyrinth Environment
    Multi-agent labyrinth env with multiple goals and zones.

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
    """

    def __init__(
        self,
        num_agents: int = 3,
        p_intended_action: float = 0.95,
        init_pos: tuple[tuple[int, int], ...] = [(1, 3), (1, 4), (1, 5)],
        subtask_config: list[SubtaskConfig] = subtask_config,
        obj_group_config: list[ObjectGroupConfig] = obj_group_config,
        detector_config: list[DetectorConfig] = detector_config,
        reward_config: RewardConfig = reward_config,
        observation_option: Literal[
            "final_goal", "intermediate_goal"
        ] = "intermediate_goal",
        width: int = 10,
        height: int = 9,
        actions_set: type[ActionsT] = NavigationActions,
        agent_dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        world: WorldT = LabyrinthWorld,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        object_options: dict[str, WorldObjT] = {
            "goal": AgentGoal,
            "door": Door,
            "zone": Zone,
            "wall": Wall,
        },
    ) -> None:
        """
        Constructor for the LabyrinthEnv class.

        Parameters
        ----------
        num_agents : int = 3
            Number of agents in the environment.
        p_intended_action : float = 0.95
            Probability of the intended action.
            Should be in the range [0, 1].
        init_pos : tuple[tuple[int, int],...] = [(1, 3), (1, 4), (1, 5)]
            Initial positions of the agents.
        subtask_config : list[SubtaskConfig] = subtask_config
            Configuration of the subtasks.
            The following keys are required:
            - "next_subtask": int | "terminal" # Next subtask index or "terminal"
            - "goal_group_index": int # Goal group index
            - "assigned_agent_goal": dict[int, tuple[int, int]] # Assigned agent goal positions
            - "triggers": list[TriggerConfig] # Triggers to open the door
        obj_group_config : list[ObjectGroupConfig] = obj_group_config
            Configuration of the object groups.
            The following keys are required:
            - "obj_type": "door" | "zone" # Object type
            - "group_index": int # Group index
            - "pos": tuple[tuple[int, int], ...] # Positions of the objects
            - "color": str # Color of the objects
            - "fill_mode": "empty" | "filled" | None # Fill mode of the object
        detector_config : list[DetectorConfig] = detector_config
            Configuration of the detectors.
            The following keys are required:
            - "obj_type": "zone" # Object type
            - "group_index": int
            - "visual_detect_prob": float
            - "radio_detect_prob": float
        reward_config : RewardConfig = reward_config
            Configuration of the rewards.
            The following keys are required:
            - "reward_option": "final_goal" | "intermediate_goal" # Reward option
            - "movement_reward": float # Movement penalty for each agent if an action is not "stay"
            - "agent_on_goal_reward": float # Reward for each agent if it is on its assigned goal
            - "agent_move_away_from_goal_reward": float # Penalty for each agent if it moves away from its assigned goal though it was on it
        observation_option : Literal["final_goal", "intermediate_goal"] = "final_goal"
            Observation option.
            - "final_goal": The observation is the positions of the final goal and agents.
            - "intermediate_goal": The observation is the positions of all the goals and agents.
        width : int = 10
            Width of the grid.
        height : int = 9
            Height of the grid.
        actions_set : type[ActionsT] = NavigationActions
            Set of actions for the agents.
            By default, there are five actions: "stay", "up", "right", "down", and "left".
        agent_dir_to_vec : list[NDArray[np.int_]] = NAV_DIR_TO_VEC
            Direction vectors for the agents.
            The length of the list should be equal to the number of actions in the actions set.
        world : WorldT = LabyrinthWorld
            World for the environment.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Render mode for the environment.
        object_options : dict[str, WorldObjT] = {"goal": AgentGoal, "door": Door, "zone": Zone, "wall": Wall}
            Options for the objects that can be placed in the environment.
        """
        self.num_agents: int = num_agents
        self.p_intended_action: float = p_intended_action
        self.subtask_config: list[SubtaskConfig] = subtask_config
        self.obj_group_config: list[ObjectGroupConfig] = obj_group_config
        self.detector_config: list[DetectorConfig] = detector_config
        self.reward_config: RewardConfig = reward_config
        self.observation_option: Literal["final_goal", "intermediate_goal"] = (
            observation_option
        )
        self.init_pos: tuple[tuple[int, int], ...] = init_pos

        agent_view_size: int = None
        agents: list[Agent] = [
            Agent(world, i, agent_view_size, actions_set, agent_dir_to_vec)
            for i in range(num_agents)
        ]

        self.final_goal: tuple[tuple[int, int], ...] = ()
        self.agent_goals: dict[int, list[tuple[int, int]]] = {}

        self.subtask_dict: dict[int, Subtask] = {
            i: Subtask(**subtask_config[i]) for i in range(len(subtask_config))
        }

        self.terminal_subtasks: list[Subtask] = [
            subtask
            for subtask in self.subtask_dict.values()
            if subtask.next_subtask == "terminal"
        ]

        for subtask in self.subtask_dict.values():
            for agent_index, pos in subtask.assigned_agent_goal.items():
                if agent_index in self.agent_goals:
                    self.agent_goals[agent_index].append(pos)
                else:
                    self.agent_goals[agent_index] = [pos]

            if subtask.next_subtask == "terminal":
                self.final_goal = tuple(subtask.assigned_agent_goal.values())
                self.final_goal_group_index = subtask.goal_group_index

        uncached_object_types: list[str] = ["agent"]

        super().__init__(
            agents=agents,
            width=width,
            height=height,
            actions_set=actions_set,
            world=world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )

        self.action_space = spaces.MultiDiscrete(
            [len(self.actions) for _ in range(self.num_agents)]
        )
        self.object_options: dict[str, WorldObjT] = object_options

    def _set_observation_space(self) -> spaces.Box:
        max_x: int = self.width - 1
        max_y: int = self.height - 1

        if self.observation_option == "final_goal":
            nums_goals: list[int] = [1 for _ in range(self.num_agents)]
        elif self.observation_option == "intermediate_goal":
            nums_goals: list[int] = [
                len(agent_goals) for agent_goals in self.agent_goals.values()
            ]
        else:
            raise ValueError(f"Invalid observation option: {self.observation_option}")

        observation_space = spaces.Dict(
            {
                str(i): spaces.Box(
                    low=np.zeros(2 * (num_goals + 1)),
                    high=np.array([max_x, max_y] * (num_goals + 1)),
                    dtype=np.int_,
                )
                for i, num_goals in enumerate(nums_goals)
            }
        )

        return observation_space

    def _find_first_subtasks(self) -> list[int]:
        # 1. Construct the graph of the goal groups
        # Each tuple contains (goal_group_index, next_goal)
        nodes: list[tuple[int, int | None]] = []
        final_nodes: list[tuple[int, int | None]] = []
        for subtask_id, subtask in self.subtask_dict.items():
            if subtask.next_subtask == "terminal":
                final_nodes.append((subtask.goal_group_index, None))
            else:
                nodes.append((subtask_id, subtask.next_subtask))

        # 2. Find the first goal groups from the final goal groups
        while True:
            next_nodes: list[tuple[int, int | None]] = []
            for node in nodes:
                for final_node in final_nodes:
                    if node[1] == final_node[0]:
                        next_nodes.append(node)
                    else:
                        pass

            # Remove duplicated nodes
            next_nodes = list(set(next_nodes))

            if len(next_nodes) == 0:
                break
            else:
                final_nodes = next_nodes

        return [node[0] for node in final_nodes]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.current_subtasks: list[Subtask] = [
            self.subtask_dict[subtask_id] for subtask_id in self._find_first_subtasks()
        ]

        self.rewarded_subtasks: list[Subtask]
        if self.reward_config["reward_option"] == "final_goal":
            self.rewarded_subtasks = [
                subtask
                for subtask in self.current_subtasks
                if subtask.next_subtask == "terminal"
            ]
        elif self.reward_config["reward_option"] == "intermediate_goal":
            self.rewarded_subtasks = self.current_subtasks
        else:
            raise ValueError(
                f"Invalid reward option: {self.reward_config['reward_option']}"
            )

        obs = self._get_obs()
        info: dict[str, Any] = self._get_info()

        return obs, info

    def _gen_grid(self, width, height) -> None:
        self.grid = Grid(width, height, self.world)

        obj_group_dict: dict[str, dict[int, ObjGroupT]] = {}

        # Place objects
        for obj_group_config in self.obj_group_config:
            obj_type: str = obj_group_config["obj_type"]
            group_index: int = obj_group_config["group_index"]

            if obj_type not in obj_group_dict:
                obj_group_dict[obj_type] = {}
            else:
                pass

            obj_group_dict[obj_type][group_index] = ObjectGroup(
                object_options=self.object_options, **obj_group_config
            )
            obj_group_dict[obj_type][group_index].put_objects(self.grid, self.world)

        self.obj_group_dict = obj_group_dict
        self.init_grid: Grid = self.grid.copy()

        # Place the agents
        assert len(self.agents) == len(self.init_pos)
        for agent, pos in zip(self.agents, self.init_pos):
            self.place_agent(agent, pos)

        # Initialize detectors
        self.detectors: list[Detector] = [
            Detector(**detector_config) for detector_config in self.detector_config
        ]

    def _get_obs(self) -> NDArray[np.int_]:
        obs: dict[str, Any] = {}

        for i, agent in enumerate(self.agents):
            agent_obs: list[tuple[int, int]] = [agent.pos]

            if self.observation_option == "final_goal":
                agent_obs.append(self.final_goal[i])
            elif self.observation_option == "intermediate_goal":
                agent_obs.extend(self.agent_goals[i])
            else:
                raise ValueError(
                    f"Invalid observation option: {self.observation_option}"
                )

            obs[str(i)] = np.array(agent_obs).flatten()

        return obs

    def step(
        self,
        actions: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        actual_actions: list[int] = self._move_agents(actions)
        obs = self._get_obs()
        reward: float = self.compute_reward(np.array(actual_actions))
        terminated: bool = (
            self._agents_detected() | self._agents_reached_terminal_goal()
        )
        truncated: bool = False
        info: dict[str, Any] = self._get_info()

        return obs, reward, terminated, truncated, info

    def _move_agents(self, actions: list[int]) -> list[int]:
        """
        Move agents based on the actions.

        Parameters
        ----------
        actions : list[int]
            Actions to take.

        Returns
        -------
        actual_actions : list[int]
            Actual actions taken.
        """
        # Randomly generate the order of the agents by indices using self.np_random.
        agent_indices: list[int] = list(range(self.num_agents))
        actual_actions: list[int] = [0 for _ in range(self.num_agents)]
        self.np_random.shuffle(agent_indices)
        for i in agent_indices:
            actual_action: int = self._move_agent(actions[i], self.agents[i])
            actual_actions[i] = actual_action

        return actual_actions

    def _move_agent(self, action: int, agent: Agent) -> int:
        """
        Move agents based on the action.

        Parameters
        ----------
        action : int
            Action to take.
        agent : Agent
            Agent to move.

        Returns
        -------
        actual_action : int
            Actual action taken.
        """

        next_pos: Position

        assert agent.pos is not None

        next_pos = agent.pos + agent.dir_to_vec[action]
        available_pos: list[Position] = self._get_available_pos(agent)

        next_pos_in_available_pos: bool = False
        for pos in available_pos:
            if np.array_equal(pos, next_pos):
                next_pos_in_available_pos = True
                break
            else:
                pass

        if next_pos_in_available_pos:
            action_probs: list[float] = []
            for pos in available_pos:
                action_probs.append(
                    self.p_intended_action
                    if np.array_equal(pos, next_pos)
                    else (1 - self.p_intended_action) / (len(available_pos) - 1)
                )

            # Normalize the action probabilities
            action_probs = np.array(action_probs) / np.sum(action_probs)
            avail_pos_indices: list[int] = list(range(len(available_pos)))

            next_pos_index = self.np_random.choice(avail_pos_indices, p=action_probs)
            next_pos = available_pos[next_pos_index]

            actual_action_vec: NDArray[np.int_] = next_pos - agent.pos
            actual_action: int = np.where(
                np.all(actual_action_vec == agent.dir_to_vec, axis=1)
            )[0][0]

            next_cell: WorldObjT | None = self.grid.get(*next_pos)
            if next_cell is None:
                agent.move(next_pos, self.grid, self.init_grid, bg_color=None)
            elif next_cell.can_overlap():
                agent.move(
                    next_pos, self.grid, self.init_grid, bg_color=next_cell.bg_color
                )
            else:
                raise ValueError(
                    f"Invalid action f{action} and position f{next_pos} for agent {agent.index}. Available positions: {available_pos}"
                )
        else:
            actual_action: int = self.actions.stay

        return actual_action

    def _get_available_pos(self, agent: Agent) -> list[Position]:
        possible_pos: list[Position] = []

        for direction in agent.dir_to_vec:
            next_pos: Position = agent.pos + direction

            if (
                next_pos[0] < 0
                or next_pos[1] < 0
                or next_pos[0] >= self.width
                or next_pos[1] >= self.height
            ):
                pass
            else:
                next_cell: WorldObjT | None = self.grid.get(*next_pos)

                if self.grid.get(*next_pos) is None:
                    possible_pos.append(next_pos)
                elif next_cell.can_overlap():
                    possible_pos.append(next_pos)
                else:
                    pass

        return possible_pos

    def _is_agent_on_terminal_goal(self, pos: Position) -> bool:
        for goal_pos in self.final_goal:
            if pos[0] == goal_pos[0] and pos[1] == goal_pos[1]:
                return True

        return False

    def _is_agent_on_assigned_goal(
        self, pos: tuple[int, int], agent_index: int, considered_subtasks: list[Subtask]
    ) -> int:
        for subtask in considered_subtasks:
            if pos == subtask.assigned_agent_goal[agent_index]:
                return subtask.goal_group_index
            else:
                pass

        return -1

    def compute_reward(self, actions: NDArray[np.int_]) -> float:
        reward: float = 0.0

        # 1. Movement penalty for each agent if an action is not "stay"
        reward += self.reward_config["movement_reward"] * np.sum(
            actions != self.actions.stay
        )

        # 2. Reward for each agent if it is on its assigned goal
        agent_goal_statuses: list[int] = []
        for agent in self.agents:
            agent_goal_statuses.append(
                self._is_agent_on_assigned_goal(
                    (agent.pos[0], agent.pos[1]), agent.index, self.current_subtasks
                )
            )

        num_goaled_agents: int = 0
        for agent in self.agents:
            for subtask in self.rewarded_subtasks:
                if (
                    agent.index in subtask.assigned_agent_goal
                    and (agent.pos[0], agent.pos[1])
                    == subtask.assigned_agent_goal[agent.index]
                ):
                    num_goaled_agents += 1
                else:
                    pass

        reward += num_goaled_agents * self.reward_config["agent_on_goal_reward"]

        # 3. Penalty for each agent if it moves away from its assigned goal though it was on it
        for agent, action in zip(self.agents, actions):
            prev_pos: Position = self._get_previous_agent_pos(action, agent)
            prev_agent_goal: int = self._is_agent_on_assigned_goal(
                (prev_pos[0], prev_pos[1]), agent.index, self.rewarded_subtasks
            )
            curr_agent_goal: int = self._is_agent_on_assigned_goal(
                (agent.pos[0], agent.pos[1]), agent.index, self.rewarded_subtasks
            )

            # If the reward option is "final_goal", the penalty is given only if the agent was on the final goal.
            if prev_agent_goal != -1 and prev_agent_goal != curr_agent_goal:
                reward += self.reward_config["agent_move_away_from_goal_reward"]
            else:
                pass

        # 4. Reward for all agents if they are on their assigned goals on the same goal group and unlock the door
        # If the door is already unlocked, the reward is not given.
        for subtask in self.current_subtasks:
            all_conditions_satisfied: bool = True
            for trigger in subtask.triggers:
                if not trigger.is_condition_satisfied(self.agents, subtask):
                    all_conditions_satisfied = False
                    break
                else:
                    pass

            all_agents_on_goals: bool = True
            for agent in self.agents:
                if (
                    agent.index in subtask.assigned_agent_goal
                    and (agent.pos[0], agent.pos[1])
                    == subtask.assigned_agent_goal[agent.index]
                ):
                    pass
                else:
                    all_agents_on_goals = False
                    break

            all_conditions_satisfied = all_conditions_satisfied and all_agents_on_goals

            if all_conditions_satisfied:
                for trigger in subtask.triggers:
                    trigger.trigger_action(self.obj_group_dict, self.grid)
                    trigger.trigger_action(self.obj_group_dict, self.init_grid)

                if subtask in self.rewarded_subtasks:
                    reward += self.reward_config["all_agents_on_goal_reward"]
                else:
                    pass

                if subtask.next_subtask != "terminal":
                    self.current_subtasks = [self.subtask_dict[subtask.next_subtask]]

                if reward_config["reward_option"] == "intermediate_goal":
                    self.rewarded_subtasks = self.current_subtasks
                else:
                    pass

                break
            else:
                pass

        return reward

    def _get_previous_agent_pos(self, action: int, agent: Agent) -> Position:
        previous_pos: Position

        assert agent.pos is not None

        previous_pos = agent.pos - agent.dir_to_vec[action]

        return previous_pos

    def _agents_detected(self) -> bool:
        detected: bool = False
        for detector in self.detectors:
            if detector.detect_agents(self.agents, self.obj_group_dict, self.np_random):
                detected = True
            else:
                pass

        return detected

    def _agents_reached_terminal_goal(self) -> bool:
        for agent in self.agents:
            if agent.pos is None:
                return False
            elif not self._is_agent_on_terminal_goal(agent.pos):
                return False
            else:
                pass

        return True
