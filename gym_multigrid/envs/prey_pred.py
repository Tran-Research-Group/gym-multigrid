from typing import Literal, TypedDict, Type
import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.constants import PREY_PRED_COLORS
from gym_multigrid.core.object import Floor
from gym_multigrid.core.world import World, WorldT
from gym_multigrid.core.agent import (
    ActionsT,
    AgentT,
    NavigationActions,
    Agent,
    PolicyAgent,
)
from gym_multigrid.core.grid import Grid
from gym_multigrid.policy import AgentPolicyT
from gym_multigrid.policy.prey_pred import PREY_PRED_POLICIES
from gym_multigrid.multigrid import (
    MultiGridEnv,
    GridConfig,
    RenderingConfig,
    PartialObsConfig,
    DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG,
)
from gym_multigrid.typing import Position

PreyPredWorld = World(
    encode_dim=3,
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
    remove_dead_preys: bool
    encode_dead_prey_as: Literal["wall", "dead", "status"]


class PredatorConfig(TypedDict):
    init_pos: tuple[int, int]
    policy_type: Literal["random", "given"]


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
        The reward given to the predator for capturing a prey of this type.
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

        super().__init__(
            world=world,
            index=index,
            actions=NavigationActions,
            color="blue",
            bg_color="light_grey",
            type=prey_type["name"],
            view_size=view_size,
        )

    def reset(self, env_generator: np.random.Generator) -> None:
        super().reset()
        self.policy: AgentPolicyT = self.policy_class(
            action_set=self.actions,
            random_generator=env_generator,
        )

    def act(self, observation: NDArray) -> int:
        return self.policy.act(observation)


class Predator(Agent):
    def __init__(
        self,
        world: WorldT,
        index: int,
        pred_config: PredatorConfig,
        view_size: int | None = None,
        policy_dict: dict[str, Type[AgentPolicyT]] = PREY_PRED_POLICIES,
    ):
        self.pred_config: PredatorConfig = pred_config
        super().__init__(
            world=world,
            index=index,
            actions=NavigationActions,
            color="red",
            bg_color="light_grey",
            type="predator",
            view_size=view_size,
        )

    def reset(self, env_generator: np.random.Generator) -> None:
        super().reset()


DEFAULT_OBSERVATION_CONFIG: ObservationConfig = {
    "encode_prey_areas": True,
    "remove_dead_agents": True,
    "encode_dead_agents_as": "wall",
}

DEFAULT_PREDATOR_CONFIGS: list[PredatorConfig] = [
    {
        "init_pos": (6, 6),
        "policy_type": "given",
    },
    {
        "init_pos": (8, 8),
        "policy_type": "random",
    },
]

DEFAULT_PREY_CONFIGS: list[PreyConfig] = [
    {
        "type": "easy_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (2, 2),
    },
    {
        "type": "easy_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (8, 2),
    },
    {
        "type": "hard_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (2, 8),
    },
    {
        "type": "hard_prey",
        "territory_dims": (4, 4),
        "territory_left_top_corner": (8, 8),
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


class PreyPredEnv(MultiGridEnv):
    """
    Environment in which the predator must catch the prey.
    """

    def __init__(
        self,
        observation_options: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
        pred_configs: list[PredatorConfig] = DEFAULT_PREDATOR_CONFIGS,
        prey_configs: list[PreyConfig] = DEFAULT_PREY_CONFIGS,
        prey_types: list[PreyType] = DEFAULT_PREY_TYPES,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        verbose: bool = False,
    ):
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

        self.observation_options: ObservationConfig = observation_options
        self.pred_configs: list[PredatorConfig] = pred_configs
        self.prey_configs: list[PreyConfig] = prey_configs
        self.prey_types: list[PreyType] = prey_types

        grid_config = GridConfig(
            grid_size=15,
            action_set=NavigationActions,
            world=world,
        )
        rendering_config: RenderingConfig = {
            "render_mode": render_mode,
            "uncached_object_types": ["predator"]
            + [prey_type["name"] for prey_type in prey_types],
        }
        partial_obs_config: PartialObsConfig = DEFAULT_FULL_OBS_ENV_PARTIAL_OBS_CONFIG

        super().__init__(
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

        # 1. the first predator must have a `given` policy whose action is given by step()
        # 2. there should be only one predator with a `given` policy
        # 3. the initial positions of the predators should not overlap

        given_policy_predator_count: int = 0
        given_policy_predator_index: int | None = None
        init_positions: list[tuple[int, int]] = []

        for pred_config in pred_configs:
            if pred_config["policy_type"] == "given":
                given_policy_predator_count += 1
                given_policy_predator_index = pred_configs.index(pred_config)
            else:
                pass

            if pred_config["init_pos"] in init_positions:
                success = False
                error_messages.append(
                    f"Invalid predator config: {pred_config['init_pos']} is already occupied."
                )
            else:
                init_positions.append(pred_config["init_pos"])

        if given_policy_predator_count != 1:
            success = False
            error_messages.append(
                "Invalid predator config: There should be exactly one predator with a `given` policy."
            )
        else:
            pass

        if given_policy_predator_index != 0:
            success = False
            error_messages.append(
                "Invalid predator config: The first predator must have a `given` policy."
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
    ) -> list[AgentT]:
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

        for pred_config in pred_configs:
            predator: Predator = Predator(
                world=self.world,
                index=len(self.agents),
                pred_config=pred_config,
                view_size=5,
            )
            agents.append(predator)

        for prey_config in prey_configs:
            prey_type: PreyType = next(
                prey_type
                for prey_type in prey_types
                if prey_type["name"] == prey_config["type"]
            )
            prey: Prey = Prey(
                prey_config=prey_config,
                prey_type=prey_type,
                world=self.world,
                index=len(self.agents),
                view_size=5,
            )
            agents.append(prey)

        return agents

    def reset(self, *, seed=None, options=None):
        self._reset_gym(seed=seed)
        self._gen_grid(self.width, self.height)
        self._reset_agents()

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
        self.grid = Grid(width, height)

        # Put walls
        self.grid.wall_rect(0, 0, width, height)

        # Place prey territories
        for prey_config in self.prey_configs:
            left_top_corner: tuple[int, int] = prey_config["territory_left_top_corner"]
            territory_dims: tuple[int, int] = prey_config["territory_dims"]

            self.grid.rect_filled(
                *left_top_corner,
                *territory_dims,
                Floor(world=self.world, color="light_grey", type="prey_area"),
            )

        self.init_grid: Grid = self.grid.copy()

    def _reset_agents(self): ...
