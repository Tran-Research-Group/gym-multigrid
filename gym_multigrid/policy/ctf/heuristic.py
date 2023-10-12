import numpy as np

from gym_multigrid.policy.base import BaseAgentPolicy


class RwPolicy(BaseAgentPolicy):
    """
    Random walk policy
    """

    def __init__(self) -> None:
        super().__init__()

    def act(self, allowed_actions: list[int]) -> int:
        return np.random.choice(allowed_actions)
