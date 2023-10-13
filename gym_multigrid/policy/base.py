from abc import abstractmethod, ABC
from typing import TypeVar

AgentPolicyT = TypeVar("AgentPolicyT", bound="BaseAgentPolicy")


class BaseAgentPolicy(ABC):
    """
    Abstract class for CTF enemy policy
    """

    def __init__(self) -> None:
        super().__init__()
        self.name: str = "base"

    @abstractmethod
    def act(self) -> int:
        ...
