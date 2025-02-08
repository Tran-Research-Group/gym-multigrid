import enum
from typing import Type

from gymnasium.core import ObsType
import numpy as np

from gym_multigrid.policy.base import BaseAgentPolicy, AgentPolicyT


class RwPolicy(BaseAgentPolicy[ObsType, int]):
    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
    ):
        super().__init__(action_set, random_generator)
        self.name = "random"

    def act(self, observation):
        return self.random_generator.choice(self.action_set)


PREY_PRED_POLICIES: dict[str, Type[AgentPolicyT]] = {
    "random": RwPolicy,
}
