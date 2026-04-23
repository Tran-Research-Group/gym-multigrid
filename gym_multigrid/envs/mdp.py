import pandas as pd
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from gymnasium import spaces, Env


class MDPAgent:

    # simple agent class for a simple MDP
    def __init__(self, init_state: int) -> None:
        self.state = init_state

    def reset(self, init_state: int):
        self.state = init_state


class ProjectMDP(Env):
    # currently only hard coded for LBF env
    """Using terminology from the project scheduling literature, this MDP represents a project which consists of multiple tasks with ordering (precedence) constraints, pre-defined transitions, and state-dependent action spaces."""

    def __init__(
        self,
        num_rooms: int,
        task_type: Literal["atomic", "composed"],
        num_comms_values: int,
    ):
        super().__init__()

        self.agent = MDPAgent(init_state=0)
        self.tasks: list[tuple]
        self.init_state: int
        self.goal_state: int
        self.fail_state: int
        self.state_space: NDArray[np.int_]
        self.successor_map: dict[tuple[int, tuple], int]
        self._build_env(
            num_rooms=num_rooms, task_type=task_type, num_comms_values=num_comms_values
        )

        # n_waypoints = n_agents is a mathematically different task from !=
        # you want to use a pandas df here for easier bookkeeping
        # self.observation_space: spaces.Discrete
        # self.action_space: spaces.Discrete

        # self.df_state: pd.DataFrame
        # df_task

    def _build_env(
        self,
        num_rooms: int,
        task_type: Literal["atomic", "composed"],
        num_comms_values: int,
    ):
        """
        assume 1 set of waypoints per room
        set of atomic tasks that can advance the state in the MDP
         - clear a room of fruit
         - all agents reach a waypoint for the current room
         - composed task = "current room cleared of fruit" and "all agents reach the current room's waypoint" are True
        """

        match (num_rooms, task_type):
            case (2, "composed"):
                """
                0 -> 1 -> 2
                """
                # need to pre-define the tasks in the project MDP
                # tasks = edges in a graph
                self.tasks: list[tuple] = [(0, 1), (1, 2)]

            case _:
                raise NotImplementedError

        self.state_space = np.arange(0, len(self.tasks) + 2)
        self.init_state = int(self.state_space[0])
        self.goal_state = int(self.state_space[-2])
        self.fail_state = int(self.state_space[-1])

        # include dummy action for self-transition of absorbing states
        self.tasks.append((self.goal_state, self.goal_state))
        self.tasks.append((self.fail_state, self.fail_state))

        self.observation_space = spaces.Discrete(n=len(self.state_space))
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(n=len(self.tasks)),
                spaces.Discrete(n=num_comms_values),
            )
        )

        # comms actions need to be be discretized to n_comms_levels

        # transition probs
        self.transition_probs: pd.DataFrame
        transition_probs: list[dict] = []

        # init probs as None until we have real data
        self.successor_map = {}
        for edge in self.tasks:
            curr_state, chosen_next_state = edge
            if edge not in [
                (self.goal_state, self.goal_state),
                (self.fail_state, self.fail_state),
            ]:
                for comms_val in range(num_comms_values):
                    action = (chosen_next_state, comms_val)
                    next_states = [chosen_next_state, self.fail_state]
                    next_state_types = [
                        "goal" if chosen_next_state == self.goal_state else "normal"
                    ]
                    next_state_types += ["fail"]

                    for next_state, next_state_type in zip(
                        next_states, next_state_types
                    ):
                        self.successor_map[(curr_state, action)] = chosen_next_state
                        transition_probs.append(
                            {
                                "state": curr_state,
                                "action": action,
                                "next_state": next_state,
                                "next_state_type": next_state_type,
                                "prob": None,
                            }
                        )

            else:
                # add dummy actions for self-transition for goal state and fail state
                # dummy action for absorbing states always has a comms val of 0 since it isn't a real task
                action = (chosen_next_state, 0)

                transition_probs.append(
                    {
                        "state": curr_state,
                        "action": action,
                        "next_state": chosen_next_state,
                        "next_state_type": (
                            "goal" if chosen_next_state == self.goal_state else "fail"
                        ),
                        "prob": None,
                    }
                )

        self.transition_probs = pd.DataFrame.from_records(transition_probs)

    def step(self, action: int):
        # move agent based on action + transition function
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)
        # TODO needs to use the base env's np_random if possible to avoid issues w/ seeding
        pass

    def _set_action_space(self):
        # MDP movement actions as well as comms allocation actions
        pass

    def _set_observation_space(self):
        pass

    def render(self):
        # low priority, get other things working first
        pass
