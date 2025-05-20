from typing import Any, Literal, TypeVar
from abc import ABC
from dataclasses import dataclass

from gym_multigrid.core.grid import Grid
from gym_multigrid.core.world import WorldT
from gym_multigrid.core.object import WorldObjT, AgentGoal, Door, Zone, Wall


class ObjectGroup(ABC):
    """abstract class for a group of environment objects"""

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


@dataclass
class EnvObjectGroup:
    """configure a set of env objects
    Parameters
    ----------
    obj_type : str
        type of object
    group_index : int
        unique index for the group of objects
    pos : tuple[tuple[int, int] | tuple[int, int, int, int], ...]
        Positions of the objects.
        A tuple element can be either a tuple of two integers or a tuple of four integers.
        - (x, y): Position of the object.
        - (x, y, w, h): Position and size of the object.
    color : str
        object color
    spawned_subtask_indices : tuple[int, ...]
        tuple of subtasks where the objects appear in the environment
    fill_mode : Literal["empty", "filled"]] | None
        empty places objects at the positions in the list, filled makes a filled-in square of the objects
    """

    obj_type: str
    group_index: int
    pos: tuple[tuple[int, int] | tuple[int, int, int, int], ...]

    color: str
    fill_mode: Literal["empty", "filled"] | None
    spawned_subtask_indices: tuple[int, ...] | None = None


ObjGroupT = TypeVar("ObjGroupT", bound=ObjectGroup)
