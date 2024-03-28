from typing import TypeVar, TypedDict

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from gym_multigrid.core.agent import ActionsT, CtfActions
from gym_multigrid.policy.base import BaseAgentPolicy, ObservationT
from gym_multigrid.policy.ctf.utils import a_star
from gym_multigrid.typing import Position

ObservationDictT = TypeVar("ObservationDictT", bound=TypedDict)


class RwPolicy(BaseAgentPolicy):
    """
    Random walk policy

    Attributes:
        name: str
            Policy name
        random_generator: numpy.random.Generator
            Random number generator. Replace it with the environment's random number generator if needed.
    """

    def __init__(
        self,
        action_set: ActionsT = CtfActions,
        random_generator: Generator | None = None,
    ) -> None:
        """
        Initialize the RW policy.

        Parameters
        ----------
        action_set : gym_multigrid.core.agent.ActionsT | None
            Actions available to the agent.
        random_generator : numpy.random.Generator
            Random number generator. Replace it with the environment's random number generator if needed.
        """
        super().__init__(action_set, random_generator)
        self.name = "rw"

    def act(self, observation: ObservationT | None = None) -> int:
        return self.random_generator.integers(0, len(self.action_set))


class FightPolicy(BaseAgentPolicy):
    """
    Policy that always tries to fight

    Attributes:
        name: str
            Policy name
    """

    def __init__(
        self,
        field_map: NDArray | None = None,
        action_set: ActionsT = CtfActions,
        random_generator: Generator | None = None,
        randomness: float = 0.75,
    ) -> None:
        """
        Initialize the policy.

        Parameters
        ----------
        field_map : numpy.typing.NDArray
            Field map of the environment.
        actions : gym_multigrid.core.agent.ActionsT
            Actions available to the agent.
        randomness : float
            Probability of taking an optimal action.
        """
        super().__init__(action_set, random_generator)
        self.name = "fight"
        self.field_map: NDArray | None = field_map
        self.randomness: float = randomness

    def act(self, observation: ObservationDictT) -> int:
        """
        Determine the action to take.

        Parameters
        ----------
        observation : ObservationDictT
            Observation dictionary (typed dict from the env).

        Returns
        -------
        int
            Action to take.
        """

        # Assert the observation dict has items "red_agent" and "blue_agent"
        assert "red_agent" in observation and "blue_agent" in observation
        shortest_path = a_star(
            observation["red_agent"], observation["blue_agent"], self.field_map
        )
        optimal_loc: Position = (
            shortest_path[1] if len(shortest_path) > 1 else observation["blue_agent"]
        )

        # Determine if the agent should take the optimal action
        is_action_optimal: bool = self.random_generator.choice(
            [True, False], p=[self.randomness, 1 - self.randomness]
        )

        # If the optimal action is not taken, return a random action
        action: int
        if is_action_optimal:
            action_dir: NDArray = np.array(optimal_loc) - observation["red_agent"]

            # Convert the direction to an action
            # stay: (0,0), left: (0,-1), down: (-1,0), right: (0,1), up: (1,0)
            if np.array_equal(action_dir, np.array([0, 0])):
                action = self.action_set.stay
            elif np.array_equal(action_dir, np.array([0, -1])):
                action = self.action_set.left
            elif np.array_equal(action_dir, np.array([-1, 0])):
                action = self.action_set.down
            elif np.array_equal(action_dir, np.array([0, 1])):
                action = self.action_set.right
            elif np.array_equal(action_dir, np.array([1, 0])):
                action = self.action_set.up
            else:
                raise ValueError("Invalid direction")
        else:
            action = self.random_generator.integers(0, len(self.action_set))

        return action
