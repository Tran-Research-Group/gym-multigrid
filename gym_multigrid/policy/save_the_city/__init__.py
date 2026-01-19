import enum
from typing import Any, Type

from gymnasium.core import ObsType
import numpy as np

from gym_multigrid.policy.base import BaseAgentPolicy, AgentPolicyT
from gym_multigrid.policy.save_the_city import utils
from gym_multigrid.core.world import SaveTheCityWorld


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
    """
    Policy for Firefighter agents.

    Navigates to the nearest burning building using A* pathfinding
    and extinguishes until fire_rate = 0.
    When no valid target exists, moves away from buildings to avoid blocking,
    then stays still.
    """

    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
        epsilon: float = 0.0,
        exclude_occupied: bool = True,
    ):
        super().__init__(action_set, random_generator)
        self.name = "firefighter"
        self.epsilon = epsilon
        self.exclude_occupied = exclude_occupied
        self.current_target: tuple[int, int] | None = None
        self.current_path: list[tuple[int, int]] | None = None
        self.path_index: int = 0
        self.world = SaveTheCityWorld

    def act(self, observation: ObsType, options: dict[str, Any]) -> int:
        """
        Select action for firefighter agent.

        Parameters
        ----------
        observation : ObsType
            Full grid observation array of shape (width, height, 6)
        options : dict[str, Any]
            Must contain 'agent_pos': tuple[int, int]

        Returns
        -------
        action : int
            Action from SaveTheCityActions
        """
        agent_pos = options.get('agent_pos')
        if agent_pos is None:
            raise ValueError("FirefighterPolicy requires 'agent_pos' in options")
        agent_pos = tuple(int(x) for x in agent_pos)  # Convert to tuple of ints

        # Apply epsilon randomness
        if self.random_generator.random() < self.epsilon:
            return utils.get_valid_move_away_action(
                agent_pos, observation, self.world,
                self.action_set, self.random_generator
            )

        # Validate current target is still valid
        if self.current_target is not None:
            if not self._is_target_valid(observation, agent_pos):
                self.current_target = None
                self.current_path = None
                self.path_index = 0

        # Find new target if needed
        if self.current_target is None:
            self.current_target = utils.find_nearest_fire(
                agent_pos, observation, self.world, self.exclude_occupied
            )
            self.current_path = None
            self.path_index = 0

        # No valid target - move away from buildings or stay still
        if self.current_target is None:
            if utils.is_adjacent_to_any_building(agent_pos, observation, self.world):
                return utils.get_valid_move_away_action(
                    agent_pos, observation, self.world,
                    self.action_set, self.random_generator
                )
            return self.action_set.STILL

        # If adjacent to target, extinguish
        if utils.is_adjacent_to_target(agent_pos, self.current_target):
            return self.action_set.EXTINGUISH

        # Need to navigate to target - compute path if needed
        if self.current_path is None or self.path_index >= len(self.current_path):
            try:
                grid = utils.obs_to_grid(observation, self.world)
                self.current_path = utils.a_star(agent_pos, self.current_target, grid)
                self.path_index = 1  # Skip first position (current location)
            except ValueError:
                # No path found - move away or stay still
                self.current_target = None
                if utils.is_adjacent_to_any_building(agent_pos, observation, self.world):
                    return utils.get_valid_move_away_action(
                        agent_pos, observation, self.world,
                        self.action_set, self.random_generator
                    )
                return self.action_set.STILL

        # Follow path
        if self.path_index < len(self.current_path):
            next_pos = self.current_path[self.path_index]
            self.path_index += 1
            return utils.direction_to_action(agent_pos, next_pos, self.action_set)

        return self.action_set.STILL

    def _is_target_valid(self, observation: ObsType, agent_pos: tuple[int, int]) -> bool:
        """Check if current target is still a valid building to extinguish."""
        if self.current_target is None:
            return False

        buildings = utils.extract_buildings_from_obs(observation, self.world)

        for building in buildings:
            if building['pos'] == self.current_target:
                # Check building is alive and still on fire
                if not building['alive']:
                    return False
                if building['fire_rate'] <= 0:
                    return False

                # Check if occupied by another agent (if enabled)
                if self.exclude_occupied:
                    agents = utils.extract_agents_from_obs(observation, self.world)
                    other_positions = [a['pos'] for a in agents if tuple(a['pos']) != tuple(agent_pos)]
                    if utils.is_building_occupied(self.current_target, other_positions):
                        return False

                return True

        return False

    def reset(self) -> None:
        """Reset policy state between episodes."""
        self.current_target = None
        self.current_path = None
        self.path_index = 0


class BuilderPolicy(BaseAgentPolicy[ObsType, int]):
    """
    Policy for Builder agents.

    Navigates to the nearest incomplete, non-burning building using A* pathfinding
    and builds until the building is complete (building_state = 100).
    When no valid target exists, moves away from buildings to avoid blocking,
    then stays still.
    """

    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
        epsilon: float = 0.0,
        exclude_occupied: bool = True,
    ):
        super().__init__(action_set, random_generator)
        self.name = "builder"
        self.epsilon = epsilon
        self.exclude_occupied = exclude_occupied
        self.current_target: tuple[int, int] | None = None
        self.current_path: list[tuple[int, int]] | None = None
        self.path_index: int = 0
        self.world = SaveTheCityWorld

    def act(self, observation: ObsType, options: dict[str, Any]) -> int:
        """
        Select action for builder agent.

        Parameters
        ----------
        observation : ObsType
            Full grid observation array of shape (width, height, 6)
        options : dict[str, Any]
            Must contain 'agent_pos': tuple[int, int]

        Returns
        -------
        action : int
            Action from SaveTheCityActions
        """
        agent_pos = options.get('agent_pos')
        if agent_pos is None:
            raise ValueError("BuilderPolicy requires 'agent_pos' in options")
        agent_pos = tuple(int(x) for x in agent_pos)  # Convert to tuple of ints

        # Apply epsilon randomness
        if self.random_generator.random() < self.epsilon:
            return utils.get_valid_move_away_action(
                agent_pos, observation, self.world,
                self.action_set, self.random_generator
            )

        # Validate current target is still valid
        if self.current_target is not None:
            if not self._is_target_valid(observation, agent_pos):
                self.current_target = None
                self.current_path = None
                self.path_index = 0

        # Find new target if needed
        if self.current_target is None:
            self.current_target = utils.find_nearest_incomplete_building(
                agent_pos, observation, self.world, self.exclude_occupied
            )
            self.current_path = None
            self.path_index = 0

        # No valid target - move away from buildings or stay still
        if self.current_target is None:
            if utils.is_adjacent_to_any_building(agent_pos, observation, self.world):
                return utils.get_valid_move_away_action(
                    agent_pos, observation, self.world,
                    self.action_set, self.random_generator
                )
            return self.action_set.STILL

        # If adjacent to target, build
        if utils.is_adjacent_to_target(agent_pos, self.current_target):
            return self.action_set.BUILD

        # Need to navigate to target - compute path if needed
        if self.current_path is None or self.path_index >= len(self.current_path):
            try:
                grid = utils.obs_to_grid(observation, self.world)
                self.current_path = utils.a_star(agent_pos, self.current_target, grid)
                self.path_index = 1  # Skip first position (current location)
            except ValueError:
                # No path found - move away or stay still
                self.current_target = None
                if utils.is_adjacent_to_any_building(agent_pos, observation, self.world):
                    return utils.get_valid_move_away_action(
                        agent_pos, observation, self.world,
                        self.action_set, self.random_generator
                    )
                return self.action_set.STILL

        # Follow path
        if self.path_index < len(self.current_path):
            next_pos = self.current_path[self.path_index]
            self.path_index += 1
            return utils.direction_to_action(agent_pos, next_pos, self.action_set)

        return self.action_set.STILL

    def _is_target_valid(self, observation: ObsType, agent_pos: tuple[int, int]) -> bool:
        """Check if current target is still a valid building to work on."""
        if self.current_target is None:
            return False

        buildings = utils.extract_buildings_from_obs(observation, self.world)

        for building in buildings:
            if building['pos'] == self.current_target:
                # Check building is alive, incomplete, and not on fire
                if not building['alive']:
                    return False
                if building['building_state'] >= 100:
                    return False
                if building['fire_rate'] > 0:
                    return False

                # Check if occupied by another agent (if enabled)
                if self.exclude_occupied:
                    agents = utils.extract_agents_from_obs(observation, self.world)
                    other_positions = [a['pos'] for a in agents if tuple(a['pos']) != tuple(agent_pos)]
                    if utils.is_building_occupied(self.current_target, other_positions):
                        return False

                return True

        return False

    def reset(self) -> None:
        """Reset policy state between episodes."""
        self.current_target = None
        self.current_path = None
        self.path_index = 0


class GeneralistPolicy(BaseAgentPolicy[ObsType, int]):
    """
    Policy for Generalist agents.

    Navigates to the nearest building that needs attention (burning OR incomplete)
    using A* pathfinding. Prioritizes firefighting over building.
    When adjacent, dynamically chooses EXTINGUISH if fire_rate > 0, else BUILD.
    When no valid target exists, moves away from buildings to avoid blocking,
    then stays still.
    """

    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = None,
        random_generator: np.random.Generator | None = None,
        epsilon: float = 0.0,
        exclude_occupied: bool = True,
    ):
        super().__init__(action_set, random_generator)
        self.name = "generalist"
        self.epsilon = epsilon
        self.exclude_occupied = exclude_occupied
        self.current_target: tuple[int, int] | None = None
        self.current_path: list[tuple[int, int]] | None = None
        self.path_index: int = 0
        self.world = SaveTheCityWorld

    def act(self, observation: ObsType, options: dict[str, Any]) -> int:
        """
        Select action for generalist agent.

        Parameters
        ----------
        observation : ObsType
            Full grid observation array of shape (width, height, 6)
        options : dict[str, Any]
            Must contain 'agent_pos': tuple[int, int]

        Returns
        -------
        action : int
            Action from SaveTheCityActions
        """
        agent_pos = options.get('agent_pos')
        if agent_pos is None:
            raise ValueError("GeneralistPolicy requires 'agent_pos' in options")
        agent_pos = tuple(int(x) for x in agent_pos)  # Convert to tuple of ints

        # Apply epsilon randomness
        if self.random_generator.random() < self.epsilon:
            return utils.get_valid_move_away_action(
                agent_pos, observation, self.world,
                self.action_set, self.random_generator
            )

        # Validate current target is still valid
        if self.current_target is not None:
            if not self._is_target_valid(observation, agent_pos):
                self.current_target = None
                self.current_path = None
                self.path_index = 0

        # Find new target if needed (prioritize fires)
        if self.current_target is None:
            target_pos, _ = utils.find_nearest_target_building(
                agent_pos, observation, self.world,
                prioritize_fire=True, exclude_occupied=self.exclude_occupied
            )
            self.current_target = target_pos
            self.current_path = None
            self.path_index = 0

        # No valid target - move away from buildings or stay still
        if self.current_target is None:
            if utils.is_adjacent_to_any_building(agent_pos, observation, self.world):
                return utils.get_valid_move_away_action(
                    agent_pos, observation, self.world,
                    self.action_set, self.random_generator
                )
            return self.action_set.STILL

        # If adjacent to target, choose action based on building state
        if utils.is_adjacent_to_target(agent_pos, self.current_target):
            return self._get_action_for_building(observation)

        # Need to navigate to target - compute path if needed
        if self.current_path is None or self.path_index >= len(self.current_path):
            try:
                grid = utils.obs_to_grid(observation, self.world)
                self.current_path = utils.a_star(agent_pos, self.current_target, grid)
                self.path_index = 1  # Skip first position (current location)
            except ValueError:
                # No path found - move away or stay still
                self.current_target = None
                if utils.is_adjacent_to_any_building(agent_pos, observation, self.world):
                    return utils.get_valid_move_away_action(
                        agent_pos, observation, self.world,
                        self.action_set, self.random_generator
                    )
                return self.action_set.STILL

        # Follow path
        if self.path_index < len(self.current_path):
            next_pos = self.current_path[self.path_index]
            self.path_index += 1
            return utils.direction_to_action(agent_pos, next_pos, self.action_set)

        return self.action_set.STILL

    def _get_action_for_building(self, observation: ObsType) -> int:
        """Determine whether to EXTINGUISH or BUILD based on building state."""
        buildings = utils.extract_buildings_from_obs(observation, self.world)

        for building in buildings:
            if building['pos'] == self.current_target:
                # Prioritize extinguishing fires
                if building['fire_rate'] > 0:
                    return self.action_set.EXTINGUISH
                # Otherwise build if incomplete
                elif building['building_state'] < 100:
                    return self.action_set.BUILD

        # Fallback - shouldn't reach here if target is valid
        return self.action_set.STILL

    def _is_target_valid(self, observation: ObsType, agent_pos: tuple[int, int]) -> bool:
        """Check if current target is still a valid building to work on."""
        if self.current_target is None:
            return False

        buildings = utils.extract_buildings_from_obs(observation, self.world)

        for building in buildings:
            if building['pos'] == self.current_target:
                # Check building is alive
                if not building['alive']:
                    return False
                # Valid if on fire OR incomplete
                if building['fire_rate'] > 0 or building['building_state'] < 100:
                    # Check if occupied by another agent (if enabled)
                    if self.exclude_occupied:
                        agents = utils.extract_agents_from_obs(observation, self.world)
                        other_positions = [a['pos'] for a in agents if tuple(a['pos']) != tuple(agent_pos)]
                        if utils.is_building_occupied(self.current_target, other_positions):
                            return False
                    return True

        return False

    def reset(self) -> None:
        """Reset policy state between episodes."""
        self.current_target = None
        self.current_path = None
        self.path_index = 0


# Policy registry for Save the City environment
SAVE_THE_CITY_POLICIES: dict[str, Type[AgentPolicyT]] = {
    "random": RandomPolicy,
    "firefighter": FirefighterPolicy,
    "builder": BuilderPolicy,
    "generalist": GeneralistPolicy,
}
