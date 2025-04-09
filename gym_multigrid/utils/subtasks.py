from abc import ABC
from typing import Callable, Tuple, Any, Literal, TypeVar
from dataclasses import dataclass

from ..core.grid import Grid
from ..core.world import WorldT
from ..core.object import WorldObjT
from ..core.object import AgentGoal, Door, Zone, Wall

import pdb


@dataclass
class EnvObjectGroup:
    """configure a set of env objects"""

    obj_type: str
    group_idx: int
    pos: tuple[tuple[int, int], ...]
    color: str
    spawned_subtask_idxs: tuple[int, ...]
    fill_mode: Literal["empty", "filled"] | None


@dataclass
class SubtaskData:
    """manages HLMDP subtask data"""

    # directed edge, defines predecessor and successor state to this subtask
    edge: Tuple[int, int]
    idx: int
    final_state: Tuple[Tuple[int, int], ...]
    termination_condition: str

    # stuff that only needs to be specified when interfacing with a gymnasium env
    init_state_dist: Tuple[Tuple[float, Tuple[Tuple[int, int], ...]], ...] | None = None
    reward_function: Callable | None = None
    termination_function: Callable | None = None
    env_object_groups: list[EnvObjectGroup] | None = None


@dataclass
class StateData:
    """manages HLMDP state data"""

    # data for a single HLMDP state
    idx: int
    # example with a single initial state: [1.0, [[1, 2], [2, 1]]]
    outgoing_init_state_dist: Tuple[Tuple[float, Tuple[Tuple[int, int], ...]], ...]
    # incoming_subtask_idxs: list | None = None
    # outgoing_subtask_idxs: list | None = None

    # not necessary for training in independent subtasks
    # def __post_init__(self):
    #     # it is bad practice to have lists as default function args (say, in __init__), so we do that here instead
    #     # __post_init__ is called after everything else in __init__ for dataclasses
    #     self.incoming_subtask_idxs = []
    #     self.outgoing_subtask_idxs = []


@dataclass
class HLMDPData:
    """manages HLMDP state and subtask data"""

    state_data_list: Tuple[StateData, ...]
    subtask_data_list: Tuple[SubtaskData, ...]

    state_data: dict[int:StateData] | None = None
    subtask_data: dict[int:SubtaskData] | None = None

    def __post_init__(self):
        self._build_data_dicts()
        # self._add_edge_data()
        self._add_init_state_dist_data()

    def _build_data_dicts(self):
        self.state_data = {state.idx: state for state in self.state_data_list}
        self.subtask_data = {subtask.idx: subtask for subtask in self.subtask_data_list}

    # def _add_edge_data(self):
    #     """adds outgoing and incoming edges to the state data
    #     """
    #     for subtask in self.subtask_data:
    #         predecessor_state_idx = subtask.edge[0]
    #         successor_state_idx = subtask.edge[1]

    #         for state in self.state_data:
    #             if state.idx == predecessor_state_idx:
    #                 state.outgoing_subtask_idxs.append(subtask.idx)
    #             elif state.idx == successor_state_idx:
    #                 state.incoming_subtask_idxs.append(subtask.idx)

    def _add_init_state_dist_data(self):
        # adds initial state distribution data to the subtask data based on its outgoing edge
        for _, data in self.subtask_data.items():
            outgoing_state_idx = data.edge[0]
            data.init_state_dist = self.state_data[
                outgoing_state_idx
            ].outgoing_init_state_dist


class ObjectGroup(ABC):
    def __init__(
        self,
        obj_type: str,
        group_idx: int,
        pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...],
        spawned_subtask_idxs: tuple[int, ...],
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
        group_idx : int
            Group index of the object.
        spawned_subtask_idxs: tuple[int, ...]
            Tuple of ints that tell which subtasks this group of objects will spawn during
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
        self.group_idx: int = group_idx
        self.spawned_subtask_idxs = spawned_subtask_idxs
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


"""
##################
# miki's stuff (old)
##################


##################
# generic classes to define subtasks, rewards, subtask completion triggers, etc.
##################
class TriggerConfig(TypedDict):
    condition: str
    action: str
    obj_type: str
    obj_group: int


class SubtaskConfig(TypedDict):
    next_subtask: int | Literal["terminal"]
    goal_group_idx: int
    assigned_agent_goal: dict[int, tuple[int, int]]
    triggers: list[TriggerConfig]


class ObjectGroupConfig(TypedDict):
    obj_type: str
    group_idx: int
    pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...]
    color: str
    fill_mode: Literal["empty", "filled"] | None


class DetectorConfig(TypedDict):
    obj_type: str
    group_idx: int
    visual_detect_prob: float
    radio_detect_prob: float


class RewardConfig(TypedDict):
    reward_option: Literal["final_goal", "intermediate_goal"]
    movement_reward: float
    agent_on_goal_reward: float
    agent_move_away_from_goal_reward: float
    all_agents_on_goal_reward: float



class Subtask:
    def __init__(
        self,
        next_subtask: int | Literal["terminal"],
        goal_group_idx: int,
        assigned_agent_goal: dict[int, tuple[int, int]],
        triggers: list[TriggerConfig],
    ) -> None:
        self.next_subtask: int | Literal["terminal"] = next_subtask
        self.goal_group_idx: int = goal_group_idx
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
        for agent_idx, goal_pos in subtask.assigned_agent_goal.items():
            if (
                agents[agent_idx].pos[0] != goal_pos[0]
                or agents[agent_idx].pos[1] != goal_pos[1]
            ):
                return False
            else:
                pass

        return True


class Detector:
    def __init__(
        self,
        obj_type: str,
        group_idx: int,
        visual_detect_prob: float,
        radio_detect_prob: float,
    ) -> None:
        self.obj_type: str = obj_type
        self.group_idx: int = group_idx
        self.visual_detect_prob: float = visual_detect_prob
        self.radio_detect_prob: float = radio_detect_prob

    def detect_agents(
        self,
        agents: list[Agent],
        obj_group_dict: dict[str, dict[int, ObjGroupT]],
        random_generator: np.random.Generator,
    ) -> bool:
        obj_group: ObjGroupT = obj_group_dict[self.obj_type][self.group_idx]
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
"""
