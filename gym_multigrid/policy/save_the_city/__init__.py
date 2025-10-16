import enum
from typing import Any, Type

from gymnasium.core import ObsType
import numpy as np

from gym_multigrid.policy.base import BaseAgentPolicy, AgentPolicyT


class RandomPolicy(BaseAgentPolicy[ObsType, int]):
    """Random policy for Save the City agents."""

    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
    ):
        super().__init__(action_set, random_generator)
        self.name = "random"

    def act(self, observation: ObsType, options: dict[str, Any]) -> int:
        """Select a random action from the action set."""
        return self.random_generator.choice(list(map(int, self.action_set)))

    def reset(self) -> None:
        """Reset policy state (no state for random policy)."""
        pass


class FirefighterPolicy(BaseAgentPolicy[ObsType, int]):
    """Policy for Firefighter agents - logic to be implemented later."""

    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
    ):
        super().__init__(action_set, random_generator)
        self.name = "firefighter"

    def act(self, observation: ObsType, options: dict[str, Any]) -> int:
        """
        Select action for firefighter agent.

        Logic will be provided by user later.
        For now, raises NotImplementedError as a placeholder.
        """
        raise NotImplementedError(
            "FirefighterPolicy logic not yet implemented. "
            "This will be filled in with specific firefighter behavior."
        )

    def reset(self) -> None:
        """Reset policy state."""
        # Will be implemented when policy logic is provided
        pass


class BuilderPolicy(BaseAgentPolicy[ObsType, int]):
    """Policy for Builder agents - logic to be implemented later."""

    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
    ):
        super().__init__(action_set, random_generator)
        self.name = "builder"

    def act(self, observation: ObsType, options: dict[str, Any]) -> int:
        """
        Select action for builder agent.

        Logic will be provided by user later.
        For now, raises NotImplementedError as a placeholder.
        """
        raise NotImplementedError(
            "BuilderPolicy logic not yet implemented. "
            "This will be filled in with specific builder behavior."
        )

    def reset(self) -> None:
        """Reset policy state."""
        # Will be implemented when policy logic is provided
        pass


class GeneralistPolicy(BaseAgentPolicy[ObsType, int]):
    """Policy for Generalist agents - logic to be implemented later."""

    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
    ):
        super().__init__(action_set, random_generator)
        self.name = "generalist"

    def act(self, observation: ObsType, options: dict[str, Any]) -> int:
        """
        Select action for generalist agent.

        Logic will be provided by user later.
        For now, raises NotImplementedError as a placeholder.
        """
        raise NotImplementedError(
            "GeneralistPolicy logic not yet implemented. "
            "This will be filled in with specific generalist behavior."
        )

    def reset(self) -> None:
        """Reset policy state."""
        # Will be implemented when policy logic is provided
        pass


# Policy registry for Save the City environment
SAVE_THE_CITY_POLICIES: dict[str, Type[AgentPolicyT]] = {
    "random": RandomPolicy,
    "firefighter": FirefighterPolicy,
    "builder": BuilderPolicy,
    "generalist": GeneralistPolicy,
}
