from collections.abc import Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PositionDist:
    """
    Discrete distribution over states

    Parameters
    ----------
    states : list[tuple[int, int]]
        States
    probs : list[float]
        Probabilities of each state
    """

    states: list[tuple[tuple[int, int], ...]]
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.probs)

        if len(self.states) > 0:
            # use allclose instead of == due to error
            # in floating point calculations
            assert np.allclose(np.sum(self.probs), 1)


@dataclass
class SubtaskData:
    """manages HLMDP subtask data
    edge: tuple[int, int]
        edge in the HLMDP that this subtask is associated with
    idx: int
        unique index for this subtask
    final_state: tuple[tuple[int, int], ...]
        final state the agents reach to complete the subtask
    termination_condition: Literal["reach_assigned_goal_state", "reach_goal_state_set"]
        possible conditions to end the subtask.
        "reach_assigned_goal_state" constructs a dict with keys as agent indices and values as the final state set for that agent, where that set only consists of a single state
        "reach_goal_state_set" is similar, except it allows any agent to be in any of the final states in the specified set
    init_state_dist: tuple[tuple[float, tuple[tuple[int, int], ...]], ...] | None
        initial state distribution for this subtask
    """

    # directed edge, defines predecessor and successor state to this subtask
    edge: tuple[int, int]
    idx: int

    # stuff that only needs to be specified when interfacing with a gymnasium env
    termination_condition: str | None = None
    goal_state_set: tuple[tuple[int, int], ...] | None = None
    init_state_dist: PositionDist | None = None


@dataclass
class StateData:
    """Manages HLMDP state data"""

    # data for a single HLMDP state
    idx: int
    outgoing_init_state_dist: PositionDist


@dataclass
class HLMDPConfig:
    """Manages HLMDP state and subtask data"""

    state_data_tuple: tuple[StateData, ...]
    subtask_data_tuple: tuple[SubtaskData, ...]

    state_data: dict[int, StateData] = field(init=False)
    subtask_data: dict[int, SubtaskData] = field(init=False)

    def __post_init__(self) -> None:
        self._build_data_dicts()
        self._add_init_state_dist_data()

    def _build_data_dicts(self) -> None:
        self.state_data: dict[int, StateData] = {
            state.idx: state for state in self.state_data_tuple
        }
        self.subtask_data: dict[int, SubtaskData] = {
            subtask.idx: subtask for subtask in self.subtask_data_tuple
        }

    def _add_init_state_dist_data(self) -> None:
        # adds initial state distribution data to the subtask data based on its outgoing edge
        for _, data in self.subtask_data.items():
            outgoing_state_index = data.edge[0]
            data.init_state_dist = self.state_data[
                outgoing_state_index
            ].outgoing_init_state_dist


def get_initial_hlmdp_config_function(env: str) -> Callable:
    """
    Get the correct get_initial_hlmdp_config function for the current environment

    Parameters
    ----------
    env : str
        name of the environment
    """
    match env:
        case "two_agent_two_task_small-v0":
            from gym_multigrid.envs.two_agent_two_task_small import get_initial_hlmdp_config
        case "two_agent_five_task_small-v0":
            from gym_multigrid.envs.two_agent_five_task_small import get_initial_hlmdp_config
        case "five_task_team_navigation-v0":
            from gym_multigrid.envs.five_task_team_navigation import get_initial_hlmdp_config
        case _:
            raise ValueError(f"{env} may not have a get_initial_hlmdp_config function or may not be included in gym-multigrid's subtasks.py file")

    return get_initial_hlmdp_config

