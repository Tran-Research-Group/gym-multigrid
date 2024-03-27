import enum
from abc import abstractmethod, ABC
from typing import TypeVar, TypedDict, Type

AgentPolicyT = TypeVar("AgentPolicyT", bound="BaseAgentPolicy")
ObservationT = TypeVar("ObservationT")


class BaseAgentPolicy(ABC):
    """
    Abstract class for CTF enemy policy
    """

    def __init__(self) -> None:
        super().__init__()
        self.name: str = "base"

    @abstractmethod
    def act(self, observation: ObservationT, actions: Type[enum.IntEnum]) -> int: ...
