import numpy as np
from numpy.typing import NDArray
from typing import TypedDict, Literal, Type

from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import SaveTheCityWorld
from gym_multigrid.core.agent import SaveTheCityActions, Firefighter, Builder, Generalist, SaveTheCityAgent
from gym_multigrid.core.object import Building, WorldObjT
from gym_multigrid.core.grid import Grid
from gym_multigrid.typing import Position
from gym_multigrid.policy.save_the_city import SAVE_THE_CITY_POLICIES


class AgentConfig(TypedDict, total=False):
    """
    Configuration for a Save the City agent.

    Attributes
    ----------
    agent_type : Literal["firefighter", "builder", "generalist"]
        The type of agent
    policy_type : Literal["ego", "teammate"]
        Whether the agent is controlled by user (ego) or policy (teammate)
    policy_name : str | None
        Name of the policy to use (None for ego agents, or a policy name from SAVE_THE_CITY_POLICIES)
    """
    agent_type: Literal["firefighter", "builder", "generalist"]
    policy_type: Literal["ego", "teammate"]
    policy_name: str | None


# Default configuration: one ego firefighter
DEFAULT_AGENT_CONFIGS: list[AgentConfig] = [
    {
        "agent_type": "firefighter",
        "policy_type": "ego",
        "policy_name": None,
    }
]


class SaveTheCityEnv(MultiGridEnv):
    """
    Environment in which agents must build buildings and extinguish fires.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30  # or 15/60/etc depending on your use case
    }

    def __init__(
        self,
        size: int = 10,
        num_buildings: int = 3,
        agent_configs: list[AgentConfig] | None = None,
        actions_set=SaveTheCityActions,
        *args,
        **kwargs
    ):
        """
        Initialize the SaveTheCity environment.

        Parameters
        ----------
        size : int
            Grid size (square).
        num_buildings : int
            Total number of buildings in the environment.
        agent_configs : list[AgentConfig] | None
            List of agent configurations specifying agent type and policy.
            If None, uses DEFAULT_AGENT_CONFIGS (one ego firefighter).
        actions_set : Enum
            Action space enum.
        respawn_fires : bool
            Whether fires randomly reignite (default: False).
        partial_obs : bool
            Whether agents have partial observability (default: False).
        """
        self.size = size
        self.num_buildings = num_buildings
        self.world = SaveTheCityWorld
        self.actions_set = actions_set
        self.respawn_fires = kwargs.get("respawn_fires", False)
        self.partial_obs: bool = kwargs.get("partial_obs", False)

        # Use default configuration if not provided
        if agent_configs is None:
            agent_configs = DEFAULT_AGENT_CONFIGS

        # Validate that at least one ego agent exists
        ego_count = sum(1 for config in agent_configs if config["policy_type"] == "ego")
        if ego_count == 0:
            raise ValueError(
                "At least one ego agent is required. "
                "Set policy_type='ego' and policy_name=None for ego agents."
            )

        # Store configurations
        self.agent_configs = agent_configs

        # Separate ego and non-ego agent indices
        self.ego_agent_indices = [
            i for i, config in enumerate(agent_configs)
            if config["policy_type"] == "ego"
        ]
        self.non_ego_agent_indices = [
            i for i, config in enumerate(agent_configs)
            if config["policy_type"] == "teammate"
        ]

        # Initialize agents based on configurations
        self.agents = self._create_agents(agent_configs)

        # Initialize per-agent tracking info
        self.info = {
            f"agent{i+1}": {"fires_extinguished": 0, "buildings_completed": 0}
            for i in range(len(self.agents))
        }

        super().__init__(
            grid_size=self.size,
            width=None,
            height=None,
            max_steps=100,
            world=self.world,
            see_through_walls=False,
            agents=self.agents,
            partial_obs=self.partial_obs,
            actions_set=self.actions_set,
            render_mode="human",
        )

    def _create_agents(self, agent_configs: list[AgentConfig]) -> list[SaveTheCityAgent]:
        """
        Create agents based on configurations.

        Parameters
        ----------
        agent_configs : list[AgentConfig]
            List of agent configurations

        Returns
        -------
        agents : list[SaveTheCityAgent]
            List of instantiated agents
        """
        agents = []
        for config in agent_configs:
            agent_type = config["agent_type"]
            policy_type = config["policy_type"]
            policy_name = config.get("policy_name")

            # Determine policy parameters
            if policy_type == "ego":
                # Ego agents have no policy
                policy_name_arg = None
                policy_dict_arg = None
            else:  # teammate
                # Teammate agents need a policy
                if policy_name is None:
                    raise ValueError(
                        f"Teammate agent of type '{agent_type}' must have a policy_name. "
                        f"Set policy_name to one of: {list(SAVE_THE_CITY_POLICIES.keys())}"
                    )
                policy_name_arg = policy_name
                policy_dict_arg = SAVE_THE_CITY_POLICIES

            # Create the agent based on type
            if agent_type == "firefighter":
                agent = Firefighter(
                    world=self.world,
                    policy_name=policy_name_arg,
                    policy_dict=policy_dict_arg,
                )
            elif agent_type == "builder":
                agent = Builder(
                    world=self.world,
                    policy_name=policy_name_arg,
                    policy_dict=policy_dict_arg,
                )
            elif agent_type == "generalist":
                agent = Generalist(
                    world=self.world,
                    policy_name=policy_name_arg,
                    policy_dict=policy_dict_arg,
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            agents.append(agent)

        return agents

    def _gen_grid(self, width: int, height: int):
        """
        Generate the grid and place buildings.
        """
        self.grid = Grid(width, height, self.world)

        # Generate surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Place buildings
        for i in range(self.num_buildings):
            # Alternate between slow and fast-burning buildings
            fast_burning = (i % 2 == 0)  # every other building is fast-burning
            burn_rate = 1

            building = Building(
                world=self.world,
                burn_rate=burn_rate,
                build_speed=2,
                firefight_speed=2,
                fast_burning=fast_burning
            )
            self.place_obj(building)

    def _reset_agents(self) -> None:
        """
        Reset the agents and place them on the grid.
        """
        # Reset agents and initialize policies BEFORE placing them
        for agent in self.agents:
            agent.reset(env_generator=self.np_random)

        # Then place agents, which sets their position and direction
        for agent in self.agents:
            self.place_agent(agent)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Reset per-agent tracking info
        self.info = {
            f"agent{i+1}": {"fires_extinguished": 0, "buildings_completed": 0}
            for i in range(len(self.agents))
        }

        # Reset step counter and grid via parent class (which handles seeding)
        # Parent's reset calls _reset_gym (sets seed), _gen_grid (builds grid), and _reset_agents (resets & places agents)
        super().reset(seed=seed, options=options)

        # Store a list of buildings for use in step()
        self.buildings = [
            obj for obj in self.grid.grid if isinstance(obj, Building)
        ]

        # Encode full-grid observation
        state = self.grid.encode()

        return state, self.info

    def _reward(
        self, current_agent: int, rewards: NDArray[np.float64], reward: float, event: str
    ) -> None:
        """
        Apply reward to the current agent and log event type.

        Parameters
        ----------
        current_agent : int
            Index of the agent to reward.
        rewards : NDArray[np.float64]
            Array of per-agent reward values.
        reward : float
            Amount of reward to give.
        event : str
            Type of event: "building_completed" or "fire_extinguished"
        """
        rewards[current_agent] += reward

        # Log the reward type in self.info
        key = f"agent{current_agent + 1}"
        if event == "building_completed":
            self.info[key]["buildings_completed"] += 1
        elif event == "fire_extinguished":
            self.info[key]["fires_extinguished"] += 1

    def move_agent(self, agent_index: int, next_cell: WorldObjT | None, next_pos: Position):
        agent = self.agents[agent_index]

        if next_cell is None or next_cell.can_overlap():
            self.grid.set(*next_pos, agent)
            self.grid.set(*agent.pos, None)
            agent.pos = next_pos


    def step(self, actions: list[int]) -> tuple[NDArray[np.int_], NDArray[np.float64], bool, bool, dict]:
        """
        Take a step in the environment.

        Parameters
        ----------
        actions : list[int]
            Actions for ego agents only. Length should equal number of ego agents.
            These come from the RL algorithm during training/evaluation.

        Returns
        -------
        obs : NDArray[np.int_]
            Observation
        reward : float
            Reward
        terminated : bool
            Whether episode is terminated
        truncated : bool
            Whether episode is truncated
        info : dict
            Additional information
        """
        # Validate that we received the correct number of ego actions
        if len(actions) != len(self.ego_agent_indices):
            raise ValueError(
                f"Expected {len(self.ego_agent_indices)} ego actions, "
                f"but received {len(actions)} actions."
            )

        # Generate actions for non-ego (teammate) agents using their policies
        obs = self.grid.encode()
        non_ego_actions = [
            self.agents[i].act(obs, {})
            for i in self.non_ego_agent_indices
        ]

        # Combine ego and non-ego actions in the correct order
        all_actions = [None] * len(self.agents)
        for ego_idx, action in zip(self.ego_agent_indices, actions):
            all_actions[ego_idx] = action
        for non_ego_idx, action in zip(self.non_ego_agent_indices, non_ego_actions):
            all_actions[non_ego_idx] = action

        # Process actions in random order (like prey_pred does)
        order = self.np_random.permutation(len(all_actions)).tolist()
        rewards = np.zeros(len(all_actions))
        terminated = False
        truncated = False
        self.step_count += 1

        for i in order:
            agent = self.agents[i]
            action = all_actions[i]

            if action in [self.actions.NORTH, self.actions.EAST, self.actions.SOUTH, self.actions.WEST]:
                for _ in range(agent.move_speed):
                    if action == self.actions.NORTH:
                        self.move_agent(i, self.grid.get(*agent.north_pos()), agent.north_pos())
                        agent.dir = 0
                    elif action == self.actions.EAST:
                        self.move_agent(i, self.grid.get(*agent.east_pos()), agent.east_pos())
                        agent.dir = 1
                    elif action == self.actions.SOUTH:
                        self.move_agent(i, self.grid.get(*agent.south_pos()), agent.south_pos())
                        agent.dir = 2
                    elif action == self.actions.WEST:
                        self.move_agent(i, self.grid.get(*agent.west_pos()), agent.west_pos())
                        agent.dir = 2

            elif action == self.actions.BUILD:
                building = self.get_nearby_building(agent)
                if building and building.fire_rate == 0:
                    result = building.build(agent.build_speed)
                    if result == "completed":
                        self._reward(i, rewards, 100, event="building_completed")

            elif action == self.actions.EXTINGUISH:
                burning = self.get_nearby_burning_building(agent)
                if burning:
                    burning.firefight_speed = agent.firefight_speed
                    burning.fight_fire()
                    if burning.fire_rate == 0:
                        self._reward(i, rewards, 20, event="fire_extinguished")


        # Let buildings update themselves (burn if on fire)
        for building in self.buildings:
            result = building.step()
            if result == "burned_down":
                # Optional: Penalize all agents (or team)
                rewards -= 50  # team penalty

        # # Filter buildings to only alive ones
        # self.buildings = [b for b in self.buildings if b.alive]

        for building in self.buildings:
            if building.alive and building.fire_rate == 0 and building.building_state < 100:
                if self.np_random.random() < 0.05:
                    building.burn()

        # Check for termination: all buildings done or burned
        terminated = bool(all(
            b.building_state == 100 or not b.alive
            for b in self.buildings
        ))
        if terminated == True:
            print("Termination condition occured!")


        if self.step_count >= self.max_steps:
            truncated = True
            print("Truncation condition occured!")

        obs = self.grid.encode()
        reward = float(np.sum(rewards))  # reduce to one float
        return obs, reward, terminated, truncated, self.info

    def get_nearby_building(self, agent):
        for pos in agent.adjacent_positions():
            obj = self.grid.get(*pos)
            if isinstance(obj, Building):
                return obj
        return None

    def get_nearby_burning_building(self, agent):
        for pos in agent.adjacent_positions():
            obj = self.grid.get(*pos)
            if isinstance(obj, Building) and obj.fire_rate > 0:
                return obj
        return None

