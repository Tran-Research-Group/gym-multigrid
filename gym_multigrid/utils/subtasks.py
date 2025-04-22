from abc import ABC
from typing import Any, Literal, TypeVar
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import pdb

from gym_multigrid.core.grid import Grid
from gym_multigrid.core.world import WorldT
from gym_multigrid.core.object import WorldObjT
from gym_multigrid.core.object import AgentGoal, Door, Zone, Wall


@dataclass
class EnvObjectGroup:
    """configure a set of env objects
        Parameters
        ----------
        obj_type : str
            type of object
        group_index : int
            unique index for the group of objects
        pos : tuple[tuple[int, int], ...]
            object position
        color : str
            object color
        spawned_subtask_indices : tuple[int, ...]
            tuple of subtasks where the objects appear in the environment
        fill_mode : Literal["empty", "filled"]] | None
            empty places objects at the positions in the list, filled makes a filled-in square of the objects
    """
    obj_type: str
    group_index: int
    pos: tuple[tuple[int, int], ...]
    color: str
    spawned_subtask_indices: tuple[int, ...]
    fill_mode: Literal["empty", "filled"] | None


@dataclass
class PositionDist:
    """
    Discrete state distribution

    Parameters
    ----------
    states : tuple[tuple[int, int], ...]
        States
    probs : list[float]
        Probabilities of each state
    """
    states: tuple[tuple[int, int], ...]
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        assert np.sum(self.probs) == 1


@dataclass
class SubtaskData:
    """manages HLMDP subtask data
        edge: tuple[int, int]
            edge in the HLMDP that this subtask is associated with
        idx: int
            unique index for this subtask
        final_state: tuple[tuple[int, int], ...]
            final state the agents reach to complete the subtask
        termination_condition: Literal["reach_assigned_final_state"]
            possible conditions to end the subtask. "reach_assigned_final_state" is the only supported condition
        init_state_dist: tuple[tuple[float, tuple[tuple[int, int], ...]], ...] | None
            initial state distribution for this subtask
    """

    # directed edge, defines predecessor and successor state to this subtask
    edge: tuple[int, int]
    idx: int
    final_state: tuple[tuple[int, int], ...]
    termination_condition: Literal["reach_assigned_final_state"]

    # stuff that only needs to be specified when interfacing with a gymnasium env
    init_state_dist: PositionDist | None = None


@dataclass
class StateData:
    """Manages HLMDP state data"""

    # data for a single HLMDP state
    idx: int
    outgoing_init_state_dist: PositionDist

    # incoming_subtask_indices: list | None = None
    # outgoing_subtask_indices: list | None = None

    # not necessary for training in independent subtasks
    # def __post_init__(self):
    #     # it is bad practice to have lists as default function args (say, in __init__), so we do that here instead
    #     # __post_init__ is called after everything else in __init__ for dataclasses
    #     self.incoming_subtask_indices = []
    #     self.outgoing_subtask_indices = []


@dataclass
class HLMDPConfig:
    """Manages HLMDP state and subtask data"""

    state_data_tuple: tuple[StateData, ...]
    subtask_data_tuple: tuple[SubtaskData, ...]

    state_data: dict[int, StateData] | None = None
    subtask_data: dict[int, SubtaskData] | None = None

    def __post_init__(self) -> None:
        self._build_data_dicts()
        # self._add_edge_data()
        self._add_init_state_dist_data()

    def _build_data_dicts(self) -> None:
        self.state_data = {state.idx: state for state in self.state_data_tuple}
        self.subtask_data = {subtask.idx: subtask for subtask in self.subtask_data_tuple}

    # def _add_edge_data(self):
    #     """adds outgoing and incoming edges to the state data
    #     """
    #     for subtask in self.subtask_data:
    #         predecessor_state_index = subtask.edge[0]
    #         successor_state_index = subtask.edge[1]

    #         for state in self.state_data:
    #             if state.idx == predecessor_state_index:
    #                 state.outgoing_subtask_indices.append(subtask.idx)
    #             elif state.idx == successor_state_index:
    #                 state.incoming_subtask_indices.append(subtask.idx)

    def _add_init_state_dist_data(self) -> None:
        # adds initial state distribution data to the subtask data based on its outgoing edge
        for _, data in self.subtask_data.items():
            outgoing_state_index = data.edge[0]
            data.init_state_dist = self.state_data[
                outgoing_state_index
            ].outgoing_init_state_dist


class ObjectGroup(ABC):
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...],
        spawned_subtask_indices: tuple[int, ...],
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
        spawned_subtask_indices: tuple[int, ...]
            tuple of ints that tell which subtasks this group of objects will spawn during
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
        self.spawned_subtask_indices = spawned_subtask_indices
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
