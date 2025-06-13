import enum
from typing import Any, Type

import numpy as np
from gymnasium.core import ObsType

from gym_multigrid.policy.base import AgentPolicy


class GivenPolicy(AgentPolicy[ObsType, int]):
    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
    ):
        super().__init__(action_set, random_generator)
        self.name = "given"

    def act(self, observation: ObsType, options: dict[str, Any]):
        return options["action"]


class RwPolicy(AgentPolicy[ObsType, int]):
    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
    ):
        super().__init__(action_set, random_generator)
        self.name = "random"

    def act(self, observation: ObsType, options: dict[str, Any]) -> int:
        if self.action_set is None:
            raise ValueError("Action set must be provided for random policy.")

        return self.random_generator.choice(list(map(int, self.action_set)))

    def reset(self) -> None:
        pass


PREY_PRED_POLICIES: dict[str, Type[AgentPolicy]] = {
    "random": RwPolicy,
}
