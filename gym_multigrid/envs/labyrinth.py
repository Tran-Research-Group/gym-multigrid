from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict, Optional

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
    action: str
    obj_type: str
    obj_group: int


class GoalGroupConfig(TypedDict):
    group_index: int
    pos: tuple[tuple[int, int], ...]
    valid_agent_indices: tuple[int, ...]
    triggers: list[TriggerConfig]
    next_goal: int | Literal["terminal"]


class ObjectGroupConfig(TypedDict):
    obj_type: str
    group_index: int
    pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...]
    color: str
    fill_mode: Literal["empty", "filled"] | None


class DetectionConfig(TypedDict):
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


goal_group_config: list[GoalGroupConfig] = [
    {
        "group_index": 0,
        "pos": ((4, 1), (4, 2), (4, 3)),
        "valid_agent_indices": (0, 1, 2),
        "triggers": [
            {
                "action": "open",
                "obj_type": "door",
                "obj_group": 0,
            }
        ],
        "next_goal": 2,
    },
    {
        "group_index": 1,
        "pos": ((3, 5), (3, 6), (3, 7)),
        "valid_agent_indices": (0, 1, 2),
        "triggers": [
            {
                "action": "open",
                "obj_type": "door",
                "obj_group": 1,
            }
        ],
        "next_goal": 2,
    },
    {
        "group_index": 2,
        "pos": ((6, 3), (6, 4), (6, 5)),
        "valid_agent_indices": (0, 1, 2),
        "triggers": [
            {
                "action": "open",
                "obj_type": "door",
                "obj_group": 2,
            }
        ],
        "next_goal": 3,
    },
    {
        "group_index": 3,
        "pos": ((8, 3), (8, 4), (8, 5)),
        "valid_agent_indices": (0, 1, 2),
        "triggers": [],
        "next_goal": "terminal",
    },
]

obj_group_config: list[ObjectGroupConfig] = [
    {
        "obj_type": "door",
        "group_index": 0,
        "pos": ((5, 1), (5, 2), (5, 3)),
        "color": "light_gray",
        "fill_mode": None,
    },
    {
        "obj_type": "door",
        "group_index": 1,
        "pos": ((4, 5), (4, 6), (4, 7)),
    },
    {
        "obj_type": "door",
        "group_index": 2,
        "pos": ((7, 2), (7, 3), (7, 4), (7, 5), (7, 6)),
    },
    {
        "obj_type": "zone",
        "group_index": 0,
        "pos": ((2, 1), (2, 2), (2, 3)),
        "group_args": {"color": "blue", "visual_detect_prob": 0.005},
    },
    {
        "obj_type": "zone",
        "group_index": 1,
        "pos": ((2, 5), (2, 6), (2, 7)),
        "group_args": {
            "color": "red",
            "visual_detect_prob": 0.005,
            "radio_detect_prob": 0.06,
        },
    },
    {
        "obj_type": "wall",
        "group_index": 0,
        "pos": ((0, 0, 10, 9),),
        "group_args": {
            "fill_mode": "empty",
        },
    },
    {
        "obj_type": "wall",
        "group_index": 1,
        "pos": ((7, 1, 2, 1), (7, 7, 2, 1), (3, 4, 2, 1)),
        "group_args": {
            "fill_mode": "filled",
        },
    },
]

detection_config: list[DetectionConfig] = [
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


class ObjectGroup(ABC):
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...],
        fill_mode: Literal["empty", "filled"] = "filled",
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

    @abstractmethod
    def _init_obj(self, world: WorldT) -> WorldObjT:
        """
        Defines the initialization of the object.
        """
        ...

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


class DoorGroup(ObjectGroup):
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...],
    ) -> None:
        super().__init__(obj_type, group_index, pos)
        self.locked: bool = True

    def _init_obj(self, world: WorldT) -> Door:
        return Door(world)

    def open(self, grid: Grid) -> None:
        for pos in self.pos:
            door: Door = grid.get(*pos)
            door.open()

        self.locked = False

    def is_locked(self) -> bool:
        return self.locked


class GoalGroup(ObjectGroup):
    def __init__(
        self,
        group_index: int,
        pos: tuple[tuple[int, int], ...],
        valid_agent_indices: tuple[int, ...],
        triggered_actions: list[str],
        next_goal: int | Literal["terminal"],
        triggered_obj_type: Optional[str] = None,
        triggered_obj_group: Optional[int] = None,
    ) -> None:
        obj_type: str = "goal"
        super().__init__(obj_type, group_index, pos)
        self.valid_agent_indices: tuple[int, ...] = valid_agent_indices
        self.triggered_actions: list[str] = triggered_actions
        self.triggered_obj_type: Optional[str] = triggered_obj_type
        self.triggered_obj_group: Optional[int] = triggered_obj_group
        self.next_goal: int | Literal["terminal"] = next_goal

    def _init_obj(self, world: WorldT, agent_index: int) -> AgentGoal:
        return AgentGoal(world, agent_index, self.group_index, color="green")

    def put_objects(self, grid: Grid, world: WorldT) -> None:
        assert len(self.pos) == len(self.valid_agent_indices)
        for pos, agent_index in zip(self.pos, self.valid_agent_indices):
            obj: AgentGoal = self._init_obj(world, agent_index)
            self._put_obj(grid, pos, obj)

    def agents_on_goals(self, agents: list[Agent]) -> bool:
        for pos, agent_index in zip(self.pos, self.valid_agent_indices):
            if (
                agents[agent_index].pos[0] != pos[0]
                or agents[agent_index].pos[1] != pos[1]
            ):
                return False
            else:
                pass

        return True

    def open(self, door_group_dict: dict[int, DoorGroup], grid: Grid) -> None:
        door_group_dict[self.triggered_obj_group].open(grid)

    def is_door_locked(self, door_group_dict: dict[int, DoorGroup]) -> bool:
        if self.triggered_obj_group is None:
            return False
        else:
            return door_group_dict[self.triggered_obj_group].is_locked()


class ZoneGroup(ObjectGroup):
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...],
        color: str,
        visual_detect_prob: float = 0.0,
        radio_detect_prob: float = 0.0,
    ) -> None:
        super().__init__(obj_type, group_index, pos)
        self.color: str = color
        self.visual_detect_prob: float = visual_detect_prob
        self.radio_detect_prob: float = radio_detect_prob

    def _init_obj(self, world: WorldT) -> Zone:
        return Zone(world, self.color, f"{self.color}_zone")

    def detect_agents(
        self, agents: list[Agent], random_generator: np.random.Generator
    ) -> bool:
        for agent in agents:
            if self.detect_agent(agent, random_generator):
                return True
            else:
                pass

        return False

    def detect_agent(self, agent: Agent, random_generator: np.random.Generator) -> bool:
        if (agent.pos[0], agent.pos[1]) in self.pos:
            visual_detect: bool = random_generator.uniform() < self.visual_detect_prob
            radio_detect: bool = random_generator.uniform() < self.radio_detect_prob
            return visual_detect or radio_detect
        else:
            return False


class WallGroup(ObjectGroup):
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...],
        fill_mode: Literal["empty", "filled"] = "filled",
    ) -> None:
        super().__init__(obj_type, group_index, pos, fill_mode)

    def _init_obj(self, world: WorldT) -> Wall:
        return Wall(world)


class ObjectGroupDict(TypedDict):
    goal: dict[int, GoalGroup]
    door: dict[int, DoorGroup]
    zone: dict[int, ZoneGroup]
    wall: dict[int, WallGroup]


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
    class RewardConfig(TypedDict):
        reward_option: Literal["final_goal", "intermediate_goal"]
        movement_reward: float
        agent_on_goal_reward: float
        agent_move_away_from_goal_reward: float
        all_agents_on_goal_reward: float

    reward_config: RewardConfig = {
        "reward_option": "final_goal",
        "movement_reward": -0.02,
        "agent_on_goal_reward": 0.2,
        "agent_move_away_from_goal_reward": -0.3,
        "all_agents_on_goal_reward": 1.0,
    }
    ```

    ## Object Groups
    - Goal objects: The goal objects are placed on the grid.
    - Door objects: The door objects are placed on the grid.
    - Zone objects: The zone objects are placed on the grid.

    ### Example
    ``` python
    goal_group_config: list[GoalGroupConfig] = [
        {
            "group_index": 0,
            "pos": ((4, 1), (4, 2), (4, 3)),
            "valid_agent_indices": (0, 1, 2),
            "triggered_actions": ["open"],
            "triggered_obj_type": "door",
            "triggered_obj_group": 0,
            "next_goal": 2,
        },
        {
            "group_index": 1,
            "pos": ((3, 5), (3, 6), (3, 7)),
            "valid_agent_indices": (0, 1, 2),
            "triggered_actions": ["open"],
            "triggered_obj_type": "door",
            "triggered_obj_group": 1,
            "next_goal": 2,
        },
        {
            "group_index": 2,
            "pos": ((6, 3), (6, 4), (6, 5)),
            "valid_agent_indices": (0, 1, 2),
            "triggered_actions": ["open"],
            "triggered_obj_type": "door",
            "triggered_obj_group": 2,
            "next_goal": 3,
        },
        {
            "group_index": 3,
            "pos": ((8, 3), (8, 4), (8, 5)),
            "valid_agent_indices": (0, 1, 2),
            "triggered_actions": ["open"],
            "triggered_obj_type": "door",
            "triggered_obj_group": -1,
            "next_goal": "terminal",
        },
    ]

    obj_group_config: list[ObjectGroupConfig] = [
        {
            "obj_type": "door",
            "group_index": 0,
            "pos": ((5, 1), (5, 2), (5, 3)),
        },
        {
            "obj_type": "door",
            "group_index": 1,
            "pos": ((4, 5), (4, 6), (4, 7)),
        },
        {
            "obj_type": "door",
            "group_index": 2,
            "pos": ((7, 2), (7, 3), (7, 4), (7, 5), (7, 6)),
        },
        {
            "obj_type": "zone",
            "group_index": 0,
            "pos": ((2, 1), (2, 2), (2, 3)),
            "group_args": {"color": "blue", "visual_detect_prob": 0.005},
        },
        {
            "obj_type": "zone",
            "group_index": 1,
            "pos": ((2, 5), (2, 6), (2, 7)),
            "group_args": {
                "color": "red",
                "visual_detect_prob": 0.005,
                "radio_detect_prob": 0.06,
            },
        },
        {
            "obj_type": "wall",
            "group_index": 0,
            "pos": ((0, 0, 10, 9),),
            "group_args": {
                "fill_mode": "empty",
            },
        },
        {
            "obj_type": "wall",
            "group_index": 1,
            "pos": ((7, 1, 2, 1), (7, 7, 2, 1), (3, 4, 2, 1)),
            "group_args": {
                "fill_mode": "filled",
            },
        },
        ]
    ```
    """

    def __init__(
        self,
        num_agents: int = 3,
        p_intended_action: float = 0.95,
        init_pos: tuple[tuple[int, int], ...] = [(1, 3), (1, 4), (1, 5)],
        goal_group_config: list[GoalGroupConfig] = goal_group_config,
        obj_group_config: list[ObjectGroupConfig] = obj_group_config,
        reward_config: RewardConfig = reward_config,
        observation_option: Literal["final_goal", "intermediate_goal"] = "final_goal",
        width: int = 10,
        height: int = 9,
        max_steps: int = 100,
        actions_set: type[ActionsT] = NavigationActions,
        agent_dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        world: WorldT = LabyrinthWorld,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
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
        goal_group_config : list[GoalGroupConfig] = goal_group_config
            Configuration of the goal groups.
            The following keys are required:
            - "group_index": int # Group index
            - "pos": tuple[tuple[int, int], ...] # Positions of the goals
            - "valid_agent_indices": tuple[int, ...] # Indices of the agents that should be on the goal
            - "triggered_actions": list[str] # Actions to call for the goal
            - "triggered_obj_type": str # Type of the object to call the action
            - "triggered_obj_group": int # Group index of the object to call the action
            - "next_goal": int | Literal["terminal"] # Next goal group index or "terminal"
        obj_group_config : list[ObjectGroupConfig] = obj_group_config
            Configuration of the object groups.
            The following keys are required:
            - "obj_type": "door" | "zone" # Object type
            - "group_index": int # Group index
            - "pos": tuple[tuple[int, int], ...] # Positions of the objects
            - "obj_args": dict[str, Any] # Arguments to initialize the object
            - "group_args": dict[str, Any] # Arguments to initialize the group and object
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
        max_steps : int = 100
            Maximum number of steps in the environment.
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
        """
        self.num_agents: int = num_agents
        self.p_intended_action: float = p_intended_action
        self.goal_group_config: list[GoalGroupConfig] = goal_group_config
        self.obj_group_config: list[ObjectGroupConfig] = obj_group_config
        self.reward_config: RewardConfig = reward_config
        self.observation_option: Literal["final_goal", "intermediate_goal"] = (
            observation_option
        )
        self.init_pos: tuple[tuple[int, int], ...] = init_pos

        agent_view_size: int = 7
        agents: list[Agent] = [
            Agent(world, i, agent_view_size, actions_set, agent_dir_to_vec)
            for i in range(num_agents)
        ]

        self.final_goal: tuple[tuple[int, int], ...] = ()
        self.goals: list[tuple[tuple[int, int], ...]] = []
        self.agent_goals: dict[int, list[tuple[int, int]]] = {}

        for goal_config in self.goal_group_config:
            self.goals.append(goal_config["pos"])
            for agent_index, pos in zip(
                goal_config["valid_agent_indices"], goal_config["pos"]
            ):
                if agent_index in self.agent_goals:
                    self.agent_goals[agent_index].append(pos)
                else:
                    self.agent_goals[agent_index] = [pos]

            if goal_config["next_goal"] == "terminal":
                self.final_goal = goal_config["pos"]
                self.final_goal_group_index = goal_config["group_index"]

        self.init_goal_group_indices: list[int] = self._find_first_goal_groups()

        uncached_object_types: list[str] = ["agent"]

        super().__init__(
            agents=agents,
            width=width,
            height=height,
            max_steps=max_steps,
            actions_set=actions_set,
            world=world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )

        self.action_space = spaces.MultiDiscrete(
            [len(self.actions) for _ in range(self.num_agents)]
        )

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

    def _find_first_goal_groups(self) -> list[int]:
        goal_group_config: list[GoalGroupConfig] = self.goal_group_config

        # 1. Construct the graph of the goal groups
        # Each tuple contains (goal_group_index, next_goal)
        nodes: list[tuple[int, int | None]] = []
        final_nodes: list[tuple[int, int | None]] = []
        for goal_group in goal_group_config:
            if goal_group["next_goal"] == "terminal":
                final_nodes.append((goal_group["group_index"], None))
            else:
                nodes.append((goal_group["group_index"], goal_group["next_goal"]))

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

        self.current_goal_group_indices: list[int] = self.init_goal_group_indices

        obs = self._get_obs()
        info: dict[str, Any] = self._get_info()

        return obs, info

    def _gen_grid(self, width, height) -> None:
        self.grid = Grid(width, height, self.world)

        obj_group_dict: ObjectGroupDict = {}

        # Place the goal objects
        obj_group_dict["goal"] = {}
        for goal_config in self.goal_group_config:
            group_index: int = goal_config["group_index"]
            positions: tuple[tuple[int, int], ...] = goal_config["pos"]

            obj_group_dict["goal"][group_index] = GoalGroup(**goal_config)

            for pos, agent_index in zip(positions, goal_config["valid_agent_indices"]):
                goal = AgentGoal(self.world, agent_index, group_index, color="green")
                self.put_obj(goal, *pos)

        # Place doors
        obj_group_dict["door"] = {}
        obj_group_dict["zone"] = {}
        obj_group_dict["wall"] = {}
        for obj_group_config in self.obj_group_config:
            obj_type: str = obj_group_config["obj_type"]
            group_index: int = obj_group_config["group_index"]

            if obj_type == "door":
                obj_group_dict[obj_type][group_index] = DoorGroup(
                    obj_type, obj_group_config["group_index"], obj_group_config["pos"]
                )

            elif obj_type == "zone":
                args: dict[str, Any] = obj_group_config["group_args"]
                obj_group_dict[obj_type][group_index] = ZoneGroup(
                    obj_type,
                    obj_group_config["group_index"],
                    obj_group_config["pos"],
                    **args,
                )

            elif obj_type == "wall":
                args: dict[str, Any] = obj_group_config["group_args"]
                obj_group_dict[obj_type][group_index] = WallGroup(
                    obj_type,
                    obj_group_config["group_index"],
                    obj_group_config["pos"],
                    **args,
                )

            else:
                raise ValueError(f"Invalid object type: {obj_type}")

            obj_group_dict[obj_type][group_index].put_objects(self.grid, self.world)

        self.obj_group_dict = obj_group_dict

        self.init_grid: Grid = self.grid.copy()

        # Place the agents
        assert len(self.agents) == len(self.init_pos)
        for agent, pos in zip(self.agents, self.init_pos):
            self.place_agent(agent, pos)

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
        truncated: bool = self.step_count >= self.max_steps
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

    def _is_agent_on_assigned_goal(self, pos: Position, agent_index: int) -> int:
        if isinstance(self.init_grid.get(*pos), AgentGoal):
            cell: AgentGoal = self.init_grid.get(*pos)
            if (
                cell.goal_group in self.current_goal_group_indices
                and cell.accepting_agent_idx == agent_index
            ):
                return cell.goal_group
            else:
                return -1
        else:
            return -1

    def compute_reward(self, actions: NDArray[np.int_]) -> float:
        reward: float = 0

        # Change the targeted goals based on the reward option
        if self.reward_config["reward_option"] == "final_goal":
            targeted_goals: list[int] = [self.final_goal_group_index]
        elif self.reward_config["reward_option"] == "intermediate_goal":
            targeted_goals: list[int] = self.current_goal_group_indices
        else:
            raise ValueError(
                f"Invalid reward option: {self.reward_config['reward_option']}"
            )

        # 1. Movement penalty for each agent if an action is not "stay"
        reward += self.reward_config["movement_reward"] * np.sum(
            actions != self.actions.stay
        )

        # 2. Reward for each agent if it is on its assigned goal
        agent_goal_statuses: list[int] = []
        for agent in self.agents:
            agent_goal_statuses.append(
                self._is_agent_on_assigned_goal(agent.pos, agent.index)
            )

        num_goaled_agents: int = 0
        for agent_goal_status in agent_goal_statuses:
            if agent_goal_status in targeted_goals:
                num_goaled_agents += 1
            else:
                pass

        reward += num_goaled_agents * self.reward_config["agent_on_goal_reward"]

        # 3. Penalty for each agent if it moves away from its assigned goal though it was on it
        for agent, action in zip(self.agents, actions):
            prev_pos: Position = self._get_previous_agent_pos(action, agent)
            prev_agent_goal: int = self._is_agent_on_assigned_goal(
                prev_pos, agent.index
            )
            curr_agent_goal: int = self._is_agent_on_assigned_goal(
                agent.pos, agent.index
            )

            # If the reward option is "final_goal", the penalty is given only if the agent was on the final goal.
            if prev_agent_goal in targeted_goals and prev_agent_goal != curr_agent_goal:
                reward += self.reward_config["agent_move_away_from_goal_reward"]
            else:
                pass

        # 4. Reward for all agents if they are on their assigned goals on the same goal group and unlock the door
        # If the door is already unlocked, the reward is not given.
        if (
            np.all(np.array(agent_goal_statuses) == agent_goal_statuses[0])
            and agent_goal_statuses[0] in self.current_goal_group_indices
        ):
            goal_group_index: int = agent_goal_statuses[0]

            if self.obj_group_dict["goal"][goal_group_index].is_door_locked(
                self.obj_group_dict["door"]
            ):
                self.obj_group_dict["goal"][goal_group_index].open(
                    self.obj_group_dict["door"], self.grid
                )
                self.obj_group_dict["goal"][goal_group_index].open(
                    self.obj_group_dict["door"], self.init_grid
                )

                self.current_goal_group_indices = [
                    self.obj_group_dict["goal"][goal_group_index].next_goal
                ]

            else:
                pass

            # The reward is given only if all the agents are on the target goal group.
            reward += (
                self.reward_config["all_agents_on_goal_reward"]
                if goal_group_index in targeted_goals
                else 0
            )

        return reward

    def _get_previous_agent_pos(self, action: int, agent: Agent) -> Position:
        previous_pos: Position

        assert agent.pos is not None

        previous_pos = agent.pos - agent.dir_to_vec[action]

        return previous_pos

    def _agents_detected(self) -> bool:
        for zone in self.obj_group_dict["zone"].values():
            if zone.detect_agents(self.agents, self.np_random):
                return True
            else:
                pass

        return False

    def _agents_reached_terminal_goal(self) -> bool:
        for agent in self.agents:
            if agent.pos is None:
                return False
            elif not self._is_agent_on_terminal_goal(agent.pos):
                return False
            else:
                pass

        return True
