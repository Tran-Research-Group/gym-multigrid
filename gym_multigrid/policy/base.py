from abc import abstractmethod, ABC


class BaseAgentPolicy(ABC):
    """
    Abstract class for CTF enemy policy
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def act(self) -> int:
        ...
