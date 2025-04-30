from typing import Literal
from dataclasses import dataclass
import numpy as np


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
        self.subtask_data = {
            subtask.idx: subtask for subtask in self.subtask_data_tuple
        }

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
