import pdb
from typing import Tuple, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class EnvObjectGroup:
    """configure a set of env objects"""

    obj_type: str
    group_idx: int
    pos: tuple[tuple[int, int], ...]
    color: str
    spawned_subtask_idxs: tuple[int, ...]


@dataclass
class RewardFunctions:
    """class to manage the set of reward functions for different subtasks"""

    termination_condition: str
    final_state: np.array

    def reward_function(
        self, agents: list, state: np.array, action: np.array, next_state: np.array
    ) -> float:
        print(agents, state, action)

        r_t = 0.0
        # get movement cost

        # get cost of other actions

        # get reward for success
        if self.termination_condition == "reach_assigned_final_state":
            return 1

        # you could add other termination conditions here

    def _assigned_state_reward_logic(self):
        pass


@dataclass
class TerminationFunctions:
    """class to manage the set of termination functions for different subtasks
    Note: this does not handle the "truncated" condition when the environment runs out of time, that must be handled separately.
    """

    termination_condition: str
    final_state: np.array

    def termination_function(
        self, agents: list, state: np.array, actions: np.array, next_state: np.array
    ) -> bool:
        terminated = 0
        if self.termination_condition == "reach_assigned_final_state":
            terminated = self._assigned_state_termination_logic(next_state)

        return terminated

    def _assigned_state_termination_logic(self, next_state: np.array) -> bool:
        # gotta update this logic to actually work for all agents with np array states
        if next_state == self.final_state:
            terminated = 1
        else:
            terminated = 0

        return terminated


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

    state_data: dict[int: StateData] | None = None
    subtask_data: dict[int: SubtaskData] | None = None

    def __post_init__(self):
        self._build_data_dicts()

    def _build_data_dicts(self):
        self.state_data = {state.idx: state for state in self.state_data_list}
        self.subtask_data = {subtask.idx: subtask for subtask in self.subtask_data_list}

    # not necessary for training in independent subtasks
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


@dataclass
class EnvData:
    state_data: dict[int: StateData]
    subtask_data: dict[int: SubtaskData]
    env_object_list: Tuple[EnvObjectGroup, ...]
    # reward_functions: RewardFunctions
    # termination_functions: TerminationFunctions

    def __post_init__(self):
        self._add_init_state_dist_data()
        self._add_env_object_data()

    def _add_init_state_dist_data(self):
        """adds initial state distribution data from states to their outgoing subtasks"""

    def _add_env_object_data(self):
        """adds environment objects to each subtask"""
        # loop over the subtasks
        for idx, data in self.subtask_data.items():
            for object_group in self.env_object_list:
                if idx in object_group.spawned_subtask_idxs:
                    if data.env_object_groups is None:
                        data.env_object_groups = []

                    data.env_object_groups.append(object_group)
