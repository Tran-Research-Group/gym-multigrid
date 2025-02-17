import enum
from typing import Any, Literal, TypedDict, Type
import warnings
import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

from gym_multigrid.core.constants import PREY_PRED_COLORS, NAV_DIR_TO_VEC
from gym_multigrid.core.object import Floor, WorldObj, WorldObjT
from gym_multigrid.core.world import World, WorldT
from gym_multigrid.core.agent import NavigationActions, Agent
from gym_multigrid.core.grid import Grid
from gym_multigrid.policy import AgentPolicyT
from gym_multigrid.policy.base import BaseAgentPolicy
from gym_multigrid.policy.prey_pred import PREY_PRED_POLICIES
from gym_multigrid.policy.prey_pred.utils import a_star
from gym_multigrid.multigrid import (
    MultiGridEnv,
    GridConfig,
    RenderingConfig,
    PartialObsConfig,
    DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG,
)

PreyPredWorld = World(
    encode_dim=2,
    normalize_obs=1,
    COLORS=PREY_PRED_COLORS,
    OBJECT_TO_IDX={
        "empty": 0,
        "wall": 1,
        "prey_area": 2,
        "predator": 3,
        "easy_prey": 4,
        "hard_prey": 5,
        "dead_prey": 6,
    },
)


class ObservationConfig(TypedDict):
    encode_prey_areas: bool
    # remove_dead_preys: bool
    # encode_dead_prey_as: Literal["wall", "dead", "status"]


class PredatorConfig(TypedDict):
    init_pos: tuple[int, int]
    policy_type: Literal["teammate", "ego"]
    color: str


class PreyType(TypedDict):
    """
    Configuration for a type of prey object in the environment.

    Attributes
    ----------
    name : str
        The name of the prey type.
    policy : Literal["random"]
        The policy used by the prey agent.
    capture_reward : float
        The reward ego to the predator for capturing a prey of this type.
    num_required_preds_capture : Literal[1, 2, 3, 4]
        The number of neighboring predators required to capture a prey of this type.
    num_required_preds_fix : Literal[1, 2, 3]
        The number of neighboring predators required to fix a prey of this type (up to a maximum of 4).
        The value should be less than or equal to `num_required_preds_capture`.
        When `num_required_preds_fix` is 1, the prey is fixed immediately after being captured.
    """

    name: str
    policy: Literal["random"]
    capture_reward: float
    num_required_preds_capture: int
    num_required_preds_fix: int


class PreyConfig(TypedDict):
    """
    Configuration for the prey agaent in the environment.

    Attributes
    ----------

    """

    type: str
    territory_dims: tuple[int, int]
    territory_left_top_corner: tuple[int, int]
    color: str


class ResetOptions(TypedDict):
    agent_pos_list: list[tuple[int, int]]


class Prey(Agent):
    def __init__(
        self,
        prey_config: PreyConfig,
        prey_type: PreyType,
        world: WorldT,
        index: int,
        view_size: int | None = None,
        policy_dict: dict[str, Type[AgentPolicyT]] = PREY_PRED_POLICIES,
    ):
        self.prey_config: PreyConfig = prey_config
        self.prey_type: PreyType = prey_type
        self.policy_class: Type[AgentPolicyT] = policy_dict[prey_type["policy"]]

        self.neighbor_pos_offsets: NDArray[np.int_] = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )
        self.neighbor_pos: NDArray[np.int_] = np.zeros((4, 2), dtype=np.int_)

        super().__init__(
            world=world,
            index=index,
            actions=NavigationActions,
            color=prey_config["color"],
            bg_color="light_grey",
            type=prey_type["name"],
            view_size=view_size,
            dir_to_vec=NAV_DIR_TO_VEC,
        )

    def get_init_pos(self) -> tuple[int, int]:
        # Randomly select a position within the prey territory
        territory_dims: tuple[int, int] = self.prey_config["territory_dims"]
        territory_left_top_corner: tuple[int, int] = self.prey_config[
            "territory_left_top_corner"
        ]

        x: int = self.policy.random_generator.integers(
            territory_left_top_corner[0],
            territory_left_top_corner[0] + territory_dims[0],
        )
        y: int = self.policy.random_generator.integers(
            territory_left_top_corner[1],
            territory_left_top_corner[1] + territory_dims[1],
        )

        return x, y

    def reset(self, env_generator: np.random.Generator) -> None:
        super().reset()
        self.policy: AgentPolicyT = self.policy_class(
            action_set=self.actions,
            random_generator=env_generator,
        )
        if self.pos is not None:
            self.neighbor_pos = self.pos + self.neighbor_pos_offsets
        else:
            self.neighbor_pos = np.zeros((4, 2), dtype=np.int_)

    def act(self, observation: NDArray, options: dict[str, Any]) -> int:
        return self.policy.act(observation, options)

    @property
    def pos(self) -> tuple[int, int]:
        return (self._pos[0], self._pos[1]) if self._pos is not None else None

    @pos.setter
    def pos(self, pos: NDArray[np.int_]) -> None:
        self._pos = pos
        if pos is not None:
            self.neighbor_pos = pos + self.neighbor_pos_offsets

    def check_pos_in_territory(self, pos: tuple[int, int]) -> bool:
        """
        Check if the given position is within the prey territory.

        Parameters
        ----------
        pos : tuple[int, int]
            The position to check.

        Returns
        -------
        in_territory : bool
            True if the position is within the prey territory, False otherwise.
        """
        territory_dims: tuple[int, int] = self.prey_config["territory_dims"]
        territory_left_top_corner: tuple[int, int] = self.prey_config[
            "territory_left_top_corner"
        ]

        in_territory: bool = (
            territory_left_top_corner[0]
            <= pos[0]
            < territory_left_top_corner[0] + territory_dims[0]
            and territory_left_top_corner[1]
            <= pos[1]
            < territory_left_top_corner[1] + territory_dims[1]
        )

        return in_territory

    def check_fix_condition(self, grid: Grid) -> bool:
        """
        Check if the prey is in a fixable condition.

        Parameters
        ----------
        grid : Grid
            The grid of the environment.

        Returns
        -------
        fixable : bool
            True if the prey is in a fixable condition, False otherwise.
        """
        num_required_preds_fix: int = self.prey_type["num_required_preds_fix"]
        num_neighbor_preds: int = 0

        for neighbor_pos in self.neighbor_pos:
            cell: None | WorldObjT = grid.get(*neighbor_pos)
            if cell is not None and cell.type == "predator":
                num_neighbor_preds += 1
            else:
                pass

        fixable: bool = num_neighbor_preds >= num_required_preds_fix

        return fixable

    def check_capture_condition(self, grid: Grid) -> bool:
        """
        Check if the prey is in a capturable condition.

        Parameters
        ----------
        grid : Grid
            The grid of the environment.

        Returns
        -------
        capturable : bool
            True if the prey is in a capturable condition, False otherwise.
        """
        num_required_preds_capture: int = self.prey_type["num_required_preds_capture"]
        num_neighbor_preds: int = 0

        for neighbor_pos in self.neighbor_pos:
            cell: None | WorldObjT = grid.get(*neighbor_pos)
            if cell is not None and cell.type == "predator":
                num_neighbor_preds += 1
            else:
                pass

        capturable: bool = num_neighbor_preds >= num_required_preds_capture

        return capturable

    def pos_in_neighbor(self, pos: tuple[int, int]) -> bool:
        """
        Check if the given position is in the prey's neighborhood.

        Parameters
        ----------
        pos : tuple[int, int]
            The position to check.

        Returns
        -------
        in_neighbor : bool
            True if the position is in the prey's neighborhood, False otherwise.
        """
        in_neighbor: bool = any(
            np.all(pos == neighbor_pos) for neighbor_pos in self.neighbor_pos
        )

        return in_neighbor


class Predator(Agent):
    def __init__(
        self,
        world: WorldT,
        index: int,
        pred_config: PredatorConfig,
        view_size: int | None = None,
    ):
        self.pred_config: PredatorConfig = pred_config

        super().__init__(
            world=world,
            index=index,
            actions=NavigationActions,
            color=pred_config["color"],
            bg_color="white",
            type="predator",
            view_size=view_size,
            dir_to_vec=NAV_DIR_TO_VEC,
        )

    def get_init_pos(self) -> tuple[int, int]:
        return self.pred_config["init_pos"]

    def reset(self, env_generator: np.random.Generator) -> None:
        super().reset()


DEFAULT_OBSERVATION_CONFIG: ObservationConfig = {
    "encode_prey_areas": True,
    # "remove_dead_agents": True,
    # "encode_dead_agents_as": "wall",
}

DEFAULT_PREDATOR_CONFIGS: list[PredatorConfig] = [
    {"init_pos": (6, 6), "policy_type": "ego", "color": "red"},
    {"init_pos": (7, 7), "policy_type": "teammate", "color": "orange"},
    {"init_pos": (8, 8), "policy_type": "teammate", "color": "yellow"},
]

DEFAULT_PREY_CONFIGS: list[PreyConfig] = [
    {
        "type": "easy_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (2, 2),
        "color": "yellow",
    },
    {
        "type": "easy_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (9, 2),
        "color": "yellow",
    },
    {
        "type": "hard_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (2, 9),
        "color": "red",
    },
    {
        "type": "hard_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (9, 9),
        "color": "red",
    },
]
DEFAULT_PREY_TYPES: list[PreyType] = [
    {
        "name": "easy_prey",
        "policy": "random",
        "capture_reward": 1.0,
        "num_required_preds_capture": 1,
        "num_required_preds_fix": 1,
    },
    {
        "name": "hard_prey",
        "policy": "random",
        "capture_reward": 1.0,
        "num_required_preds_capture": 2,
        "num_required_preds_fix": 1,
    },
]


class PreyPredEnv(MultiGridEnv[NDArray[np.int_], list[int] | NDArray[np.int_]]):
    """
    Environment in which the predator must catch the prey.

    Observation
    -----------
    - Observation is a 3D array of shape (2, width, height).
    - The first layer is the static objects (walls, prey territories).
    - The second layer is the dynamic objects (agents).
    - Encoding:
        - empty: 0
        - wall: 1
        - prey_area: 2
        - predator: 3
        - easy_prey: 4
        - hard_prey: 5

    Actions
    -------
    - 5 discrete actions:
        - 0: stay
        - 1: move left
        - 2: move down
        - 3: move right
        - 4: move up
    - Predator agents movements have to be supplied by the user via `step()`.
    - Prey agents movements are generated by the environment. Defaulted as random.

    Preys
    -----
    - Preys have territories in which they can move.
    - Preys can be of two types: easy and hard.
    - Easy preys require 1 predator to capture and 0 predator to fix (immediately captured).
    - Hard preys require 2 predators to capture and 1 predator to fix.

    """

    def __init__(
        self,
        max_episode_steps: int = 300,
        observation_config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
        pred_configs: list[PredatorConfig] = DEFAULT_PREDATOR_CONFIGS,
        prey_configs: list[PreyConfig] = DEFAULT_PREY_CONFIGS,
        prey_types: list[PreyType] = DEFAULT_PREY_TYPES,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        verbose: bool = False,
    ):
        self.max_episode_steps: int = max_episode_steps
        self.verbose: bool = verbose

        world = PreyPredWorld

        # Check if the pred & prey configurations and prey types are valid
        pred_configs_valid, pred_error_messages = self._check_pred_configs(pred_configs)
        prey_configs_valid, prey_error_messages = self._check_prey_configs(
            prey_configs, prey_types, world
        )
        if not pred_configs_valid or not prey_configs_valid:
            error_messages = pred_error_messages + prey_error_messages
            raise ValueError("\n".join(error_messages))
        else:
            pass

        self.observation_config: ObservationConfig = observation_config
        self.pred_configs: list[PredatorConfig] = pred_configs
        self.prey_configs: list[PreyConfig] = prey_configs
        self.prey_types: list[PreyType] = prey_types

        self.num_preys: int = len(prey_configs)
        self.num_preds: int = len(pred_configs)

        grid_config: GridConfig = {
            "grid_size": 15,
            "actions_set": NavigationActions,
            "world": world,
        }
        rendering_config: RenderingConfig = {
            "render_mode": render_mode,
            "uncached_object_types": ["predator"]
            + [prey_type["name"] for prey_type in prey_types],
        }
        partial_obs_config: PartialObsConfig = DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG

        agents: list[Predator | Prey] = self._gen_agents(
            pred_configs, prey_configs, prey_types, world
        )

        super().__init__(
            agents=agents,
            **grid_config,
            **rendering_config,
            **partial_obs_config,
        )

    def _check_pred_configs(
        self, pred_configs: list[PredatorConfig]
    ) -> tuple[bool, list[str]]:
        """
        Check if the predator configurations are valid.

        Parameters
        ----------
        pred_configs : list[PredatorConfig]
            The configuration for the predator agents in the environment.

        Returns
        -------
        success : bool
            True if the configuration is valid, False otherwise.
        error_messages : list[str]
            A list of error messages if the configuration is invalid.
        """

        success: bool = True
        error_messages: list[str] = []

        # 1. the first predator must have a `ego` policy whose action is ego by step()
        # 2. there should be only one predator with a `ego` policy
        # 3. the initial positions of the predators should not overlap

        ego_predator_count: int = 0
        ego_predator_index: int | None = None
        init_positions: list[tuple[int, int]] = []

        for pred_config in pred_configs:
            if pred_config["policy_type"] == "ego":
                ego_predator_count += 1
                ego_predator_index = pred_configs.index(pred_config)
            else:
                pass

            if pred_config["init_pos"] in init_positions:
                success = False
                error_messages.append(
                    f"Invalid predator config: {pred_config['init_pos']} is already occupied."
                )
            else:
                init_positions.append(pred_config["init_pos"])

        if ego_predator_count != 1:
            success = False
            error_messages.append(
                "Invalid predator config: There should be exactly one predator with a `ego` policy."
            )
        else:
            pass

        if ego_predator_index != 0:
            success = False
            error_messages.append(
                "Invalid predator config: The first predator must have a `ego` policy."
            )
        else:
            pass

        return success, error_messages

    def _check_prey_configs(
        self, prey_configs: list[PreyConfig], prey_types: list[PreyType], world: WorldT
    ) -> tuple[bool, list[str]]:
        """
        Check if the preys configuration and prey types are valid.

        Parameters
        ----------
        prey_configs : list[PreyConfig]
            The configuration for the prey agents in the environment.
        prey_types : list[PreyType]
            The configuration for the prey types in the environment.
        world : WorldT
            The world configuration.

        Returns
        -------
        success : bool
            True if the configuration is valid, False otherwise.
        error_messages : list[str]
            A list of error messages if the configuration is invalid.
        """

        success: bool = True
        error_messages: list[str] = []

        for prey_type in prey_types:
            if prey_type["name"] not in world.OBJECT_TO_IDX:
                success = False
                error_messages.append(
                    f"Invalid prey type: {prey_type['name']} not in world object to index mapping."
                )
            else:
                pass

            if (
                prey_type["num_required_preds_fix"]
                > prey_type["num_required_preds_capture"]
            ):
                success = False
                error_messages.append(
                    f"Invalid prey type: {prey_type['name']} has num_required_preds_fix greater than num_required_preds_capture."
                )
            else:
                pass

        prey_type_names = [prey_type["name"] for prey_type in prey_types]

        for prey_config in prey_configs:
            if prey_config["type"] not in prey_type_names:
                success = False
                error_messages.append(
                    f"Invalid prey config: {prey_config['type']} not in prey types."
                )

        return success, error_messages

    def _gen_agents(
        self,
        pred_configs: list[PredatorConfig],
        prey_configs: list[PreyConfig],
        prey_types: list[PreyType],
        world: WorldT,
    ) -> list[Predator | Prey]:
        """
        Generate the agents for the environment.

        Parameters
        ----------
        pred_configs : list[PredatorConfig]
            The configuration for the predator agents in the environment.
        prey_configs : list[PreyConfig]
            The configuration for the prey agents in the environment.
        prey_types : list[PreyType]
            The configuration for the prey types in the environment.
        """
        agents: list[Agent] = []

        for i, pred_config in enumerate(pred_configs):
            predator: Predator = Predator(
                world=world,
                index=i,
                pred_config=pred_config,
                view_size=None,
            )
            agents.append(predator)

        for j, prey_config in enumerate(prey_configs):
            prey_type: PreyType = next(
                prey_type
                for prey_type in prey_types
                if prey_type["name"] == prey_config["type"]
            )
            prey: Prey = Prey(
                prey_config=prey_config,
                prey_type=prey_type,
                world=world,
                index=j,
                view_size=None,
            )
            agents.append(prey)

        return agents

    def _set_observation_space(self) -> spaces.Box:
        observation_space = spaces.Box(
            low=0,
            high=len(self.world.OBJECT_TO_IDX) - 1,
            shape=(2, self.width, self.height),
            dtype=np.int_,
        )
        return observation_space

    def reset(
        self, seed: int | None = None, options: ResetOptions | None = None
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        self.step_count: int = 0

        self._reset_gym(seed=seed)
        self._gen_grid(self.width, self.height)

        # Reset the agents. If agent_pos_list is provided, use it to reset the agents
        agent_pos_list: list[tuple[int, int]] | None = None
        if options is not None:
            agent_pos_list = options.get("agent_pos_list", None)
        else:
            pass
        self._reset_agents(agent_pos_list=agent_pos_list)

        obs: NDArray[np.int_] = self._get_obs()
        info: dict[str, Any] = self._get_info()

        return obs, info

    def _get_obs(self) -> NDArray[np.int_]:
        """
        Get the observation of the environment.

        Returns
        -------
        observation : NDArray[np.int_]
            The observation of the environment.
            1st layer: static objects
            2nd layer: dynamic objects (agents)
        """

        obs: NDArray[np.int_] = np.zeros((2, self.width, self.height), dtype=np.int_)

        obs[0, :, :] = self.static_obs

        # Add agent ids to the observation
        for agent in self.agents:
            if not agent.terminated:
                obs[1, agent.pos[1], agent.pos[0]] = self.world.OBJECT_TO_IDX[
                    agent.type
                ]
            else:
                pass

        return obs

    def _get_info(self):
        return super()._get_info()

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generate the grid for the environment.

        Parameters
        ----------
        width : int
            The width of the grid.
        height : int
            The height of the grid.
        """
        self.grid = Grid(width, height, self.world)

        # Put white floors everywhere
        self.grid.rect_filled(
            0, 0, width, height, Floor(world=self.world, color="white", type="empty")
        )

        # Put walls
        self.grid.wall_rect(0, 0, width, height)

        static_obs: NDArray[np.int_] = np.zeros(
            (self.width, self.height), dtype=np.int_
        )
        # Place prey territories
        for prey_config in self.prey_configs:
            left_top_corner: tuple[int, int] = prey_config["territory_left_top_corner"]
            territory_dims: tuple[int, int] = prey_config["territory_dims"]

            self.grid.rect_filled(
                *left_top_corner,
                *territory_dims,
                Floor(world=self.world, color="light_grey", type="prey_area"),
            )

        # Add static objects ids to the observation
        for i in range(self.width):
            for j in range(self.height):
                cell: None | WorldObjT = self.grid.get(i, j)
                if cell is None:
                    static_obs[j, i] = self.world.OBJECT_TO_IDX["empty"]
                elif (
                    cell.type == "prey_area"
                    and not self.observation_config["encode_prey_areas"]
                ):
                    static_obs[j, i] = self.world.OBJECT_TO_IDX["empty"]
                else:
                    static_obs[j, i] = self.world.OBJECT_TO_IDX[cell.type]

        self.static_obs: NDArray[np.int_] = static_obs

        self.init_grid: Grid = self.grid.copy()

    def _reset_agents(
        self, agent_pos_list: list[tuple[int, int]] | None = None
    ) -> None:
        """
        Reset the agents in the environment.
        """
        for agent in self.agents:
            agent.reset(self.np_random)

        use_agent_pos_list: bool = agent_pos_list is not None
        if agent_pos_list is not None:
            if len(agent_pos_list) != len(self.agents):
                warnings.warn(
                    f"""Invalid agent position list: Expected {len(self.agents)} positions, got {len(agent_pos_list)}.
                    Resetting agents to their initial positions.""",
                    UserWarning,
                )
                use_agent_pos_list = False
            else:
                pass

        for agent in self.agents:
            agent_pos: tuple[int, int] = (
                agent.get_init_pos()
                if not use_agent_pos_list
                else agent_pos_list.pop(0)
            )
            self.place_agent(agent, agent_pos)

    def step(
        self, actions: list[int] | NDArray[np.int_]
    ) -> tuple[NDArray[np.int_], float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Parameters
        ----------
        action : int
            The action to take.

        Returns
        -------
        observation : NDArray[np.int_]
            The observation of the environment.
        reward : float
            The reward for the action.
        terminated: bool
            Whether the episode has terminated.
        terminated: bool
            Whether the episode has terminated.
        info : dict[str, Any]
            Additional information about the environment.
        """
        terminated: bool = False
        reward: float = 0

        actions = np.array(actions)
        prey_actions = [
            agent.act(self._get_obs(), {}) for agent in self.agents[self.num_preds :]
        ]
        all_actions = np.concatenate((actions, prey_actions))
        # Order to apply the actions to the agents
        order: NDArray[np.int_] = self.np_random.permutation(len(self.agents))

        for i in order:
            agent: Prey | Predator = self.agents[i]
            action: int = all_actions[i]

            if agent.terminated:
                continue
            else:
                next_pos: tuple[int, int] = self._get_next_pos(agent, action)
                next_cell: None | WorldObjT = self.grid.get(*next_pos)

                if isinstance(next_cell, WorldObj) and not next_cell.can_overlap():
                    continue
                else:
                    if isinstance(agent, Prey) and (
                        agent.check_fix_condition(self.grid)
                        or not agent.check_pos_in_territory(next_pos)
                    ):
                        continue
                    else:
                        pass

                    # Move agent
                    self.grid.set(*next_pos, agent)
                    self.grid.set(*agent.pos, self.init_grid.get(*agent.pos))
                    # Change the dir of the agent
                    if agent.pos != next_pos:
                        dir_vec: NDArray[np.int_] = np.array(next_pos) - np.array(
                            agent.pos
                        )
                        # dir_vec = (
                        #     next_pos[0] - agent.pos[0],
                        #     next_pos[1] - agent.pos[1],
                        # )
                        # for dir, vec in enumerate(agent.dir_to_vec):
                        #     if vec[0] == dir_vec[0] and vec[1] == dir_vec[1]:
                        #         agent.dir = dir
                        #         break
                        agent.dir = agent.vec2dir(dir_vec)
                    agent.pos = next_pos

                    # Update the agent's bg_color
                    init_grid_cell: WorldObjT | None = self.init_grid.get(*agent.pos)
                    if init_grid_cell is not None:
                        agent.bg_color = self.init_grid.get(*agent.pos).color

                    # Determine rewards and remove captured preys from the grid
                    for prey in self.agents[self.num_preds :]:
                        if prey.terminated:
                            continue
                        else:
                            prey_captured: bool = prey.check_capture_condition(
                                self.grid
                            )
                            if prey_captured:
                                prey.terminated = True
                                reward += prey.prey_type["capture_reward"]
                                self.grid.set(*prey.pos, self.init_grid.get(*prey.pos))
                            else:
                                continue

        # Terminate the episode if all the prey are captured
        terminated = all(prey.terminated for prey in self.agents[self.num_preds :])
        truncated = self.step_count >= self.max_episode_steps

        self.step_count += 1

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_next_pos(self, agent: Predator | Prey, action: int) -> tuple[int, int]:
        next_pos: NDArray[np.int_]
        match action:
            case self.actions.stay:
                next_pos = agent.pos
            case self.actions.left:
                next_pos = agent.west_pos()
            case self.actions.right:
                next_pos = agent.east_pos()
            case self.actions.up:
                next_pos = agent.north_pos()
            case self.actions.down:
                next_pos = agent.south_pos()
            case _:
                raise ValueError(f"Invalid action: {action}")

        next_pos: tuple[int, int] = (next_pos[0], next_pos[1])

        return next_pos

    def _move_prey(self, prey: Prey, next_pos: tuple[int, int]) -> None:
        """
        Move the prey agent to the next position.

        Parameters
        ----------
        prey : Prey
            The prey agent to move.
        next_pos : tuple[int, int]
            The next position to move the agent to.
        """
        next_cell: None | WorldObjT = self.grid.get(*next_pos)

        agent_moves: bool = False

    def _move_pred(self, pred: Predator, next_pos: tuple[int, int]) -> float:
        """
        Move the predator agent to the next position.

        Parameters
        ----------
        pred : Predator
            The predator agent to move.
        next_pos : tuple[int, int]
            The next position to move the agent to.

        Returns
        -------
        reward : float
            The reward for the action.
        """
        pass

    @property
    def prey_agents(self) -> list[Prey]:
        return self.agents[self.num_preds :]


class BasePolicy:
    def __init__(
        self,
        action_set: Type[enum.IntEnum] | None = NavigationActions,
        dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        random_generator: np.random.Generator | None = None,
    ):
        self.dir_to_vec: list[NDArray[np.int_]] = dir_to_vec
        self.action_set: Type[enum.IntEnum] = action_set
        self.random_generator: np.random.Generator = (
            random_generator
            if random_generator is not None
            else np.random.default_rng()
        )

    def act(self, action_option: dict[str, Any]) -> int:
        """
        Choose an action for the predator agent.

        Parameters
        ----------
        observation : NDArray[np.int_]
            The observation of the environment.
        options : GreedyPredatorActionOption
            Additional options for the action.

        Returns
        -------
        action : int
            The action to take.
        """
        ...

    def vec2dir(self, vec: NDArray[np.int_]) -> int:
        """
        Convert a vector to a direction.

        Parameters
        ----------
        vec : NDArray[np.int_]
            The vector to convert.

        Returns
        -------
        dir : int
            The direction corresponding to the vector.
        """
        for dir, dir_vec in enumerate(self.dir_to_vec):
            if np.array_equal(dir_vec, vec):
                return dir
        raise ValueError(f"Invalid vector: {vec}")


class GreedyPredatorActionOption(TypedDict):
    current_pos: tuple[int, int]
    preys: list[Prey]
    grid: Grid


class GreedyPredatorPolicy(BasePolicy):
    def __init__(
        self,
        target_list: list[Literal[0, 1]],
        random_prob: float = 0.1,
        action_set: Type[enum.IntEnum] | None = NavigationActions,
        dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        random_generator: np.random.Generator | None = None,
    ):
        self.target_list: list[Literal[0, 1]] = target_list
        self.random_prob: float = random_prob
        super().__init__(
            action_set=action_set,
            dir_to_vec=dir_to_vec,
            random_generator=random_generator,
        )

    def act(self, action_option: GreedyPredatorActionOption) -> int:
        """
        Choose an action for the predator agent.

        Parameters
        ----------
        observation : NDArray[np.int_]
            The observation of the environment.
        options : GreedyPredatorActionOption
            Additional options for the action.

        Returns
        -------
        action : int
            The action to take.
        """
        current_pos: tuple[int, int] = action_option["current_pos"]
        preys: list[Prey] = action_option["preys"]
        grid: Grid = action_option["grid"]

        target_prey: None | Prey = None

        for prey, target_candidate in zip(preys, self.target_list):
            if target_candidate == 1 and not prey.terminated:
                target_prey = prey
                break
            else:
                pass

        # If there is no target prey, choose a random action
        # Else, choose the greedy action to move towards the target prey with random probability `random_prob`
        act_randomly: bool = (
            False
            if target_prey is not None
            and (
                target_prey.pos_in_neighbor(current_pos)
                or self.random_generator.random() >= self.random_prob
            )
            else True
        )

        action: int

        match act_randomly:
            case True:
                action = self.random_generator.integers(0, len(self.action_set))
            case False:
                path: list[tuple[int, int]] = a_star(current_pos, target_prey.pos, grid)
                next_pos: tuple[int, int] = path[1] if len(path) > 1 else current_pos
                dir_vec: NDArray[np.int_] = np.array(next_pos) - np.array(current_pos)
                action = self.vec2dir(dir_vec)

        return action
