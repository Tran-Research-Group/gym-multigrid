from abc import abstractmethod, ABC
from typing import TypeVar, TypedDict, Type

from gym_multigrid.core.agent import ActionsT

AgentPolicyT = TypeVar("AgentPolicyT", bound="BaseAgentPolicy")


class BaseAgentPolicy(ABC):
    """
    Abstract class for CTF enemy policy
    """

    def __init__(self) -> None:
        super().__init__()
        self.name: str = "base"

    @abstractmethod
    def act(self, observation: TypedDict, actions: Type[ActionsT]) -> int:
        ...
