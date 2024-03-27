from typing import TypedDict

import numpy as np
from numpy.random import Generator

from gym_multigrid.core.agent import ActionsT
from gym_multigrid.policy.base import BaseAgentPolicy, ObservationT


class RwPolicy(BaseAgentPolicy):
    """
    Random walk policy

    Attributes:
        name: str
            Policy name
        random_generator: numpy.random.Generator
            Random number generator. Replace it with the environment's random number generator if needed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "rw"
        self.random_generator: Generator = np.random.default_rng()

    def act(self, observation: ObservationT, actions: ActionsT) -> int:
        return self.random_generator.integers(0, len(actions))
