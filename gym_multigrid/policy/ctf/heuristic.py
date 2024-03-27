from typing import TypedDict

import numpy as np

from gym_multigrid.core.agent import ActionsT
from gym_multigrid.policy.base import BaseAgentPolicy, ObservationT


class RwPolicy(BaseAgentPolicy):
    """
    Random walk policy
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "rw"

    def act(self, observation: ObservationT, actions: ActionsT) -> int:
        return np.random.choice(list(actions))
