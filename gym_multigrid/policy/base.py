import enum
from abc import ABC, abstractmethod
from typing import Any, Generic, Type

import numpy as np
from gymnasium.core import ActType, ObsType
from numpy.random import Generator


class AgentPolicy(Generic[ObsType, ActType], ABC):
    """
    Abstract class for CTF enemy policy
    """

    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: Generator | None = None,
    ) -> None:
        """
        Base class for CTF agent policy.

        Parameters
        ----------
        action_set : enum.IntEnum
            Actions available to the agent.
        random_generator : numpy.random.Generator
            Random number generator. Replace it with the environment's random number generator if needed.
        """
        super().__init__()
        self.name: str = "base"
        self.action_set: Type[enum.IntEnum] | None = action_set
        self.random_generator: Generator = (
            random_generator
            if random_generator is not None
            else np.random.default_rng()
        )

    @abstractmethod
    def act(self, observation: ObsType, options: dict[str, Any]) -> ActType: ...

    @abstractmethod
    def reset(self) -> None: ...
