import numpy as np
from numpy.typing import NDArray

from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.core.world import SaveTheCityWorld
from gym_multigrid.core.agent import SaveTheCityActions, Firefighter, Builder, Generalist
from gym_multigrid.core.object import Building, WorldObjT
from gym_multigrid.core.grid import Grid
from gym_multigrid.typing import Position


class SaveTheCityEnv(MultiGridEnv):
    """
    Environment in which agents must build buildings and extinguish fires.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30  # or 15/60/etc depending on your use case
    }

    def __init__(self, size: int = 10, num_buildings: int = 3, agent_types: list[str] = None,
                 actions_set=SaveTheCityActions, *args, **kwargs):
        """
        Initialize the SaveTheCity environment.

        Parameters
        ----------
        size : int
            Grid size (square).
        num_buildings : int
            Total number of buildings in the environment.
        agent_types : list[str]
            List of agent types: "firefighter", "builder", "generalist".
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

        # Default agent types if not provided
        if agent_types is None:
            agent_types = ["firefighter", "builder", "generalist"]

        # Initialize agents
        self.agents = []
        for agent_type in agent_types:
            if agent_type == "firefighter":
                self.agents.append(Firefighter(self.world))
            elif agent_type == "builder":
                self.agents.append(Builder(self.world))
            elif agent_type == "generalist":
                self.agents.append(Generalist(self.world))
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

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

    def _gen_grid(self, width: int, height: int):
        """
        Generate the grid and place buildings and agents.
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

        # Place agents randomly
        for agent in self.agents:
            self.place_agent(agent)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Reset random seed if needed
        if seed is not None:
            self._np_random, seed = self.seed(seed)

        # Reset per-agent tracking info
        self.info = {
            f"agent{i+1}": {"fires_extinguished": 0, "buildings_completed": 0}
            for i in range(len(self.agents))
        }

        # Reset step counter and grid via parent class
        super().reset(seed=seed, options=options)

        # Optionally store a list of buildings for use in step()
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


    def step(
        self, actions: list[int] | NDArray[np.int_]
    ) -> tuple[NDArray[np.int_], NDArray[np.float64], bool, bool, dict]:
        order: list[int] = np.random.permutation(len(actions)).tolist()
        rewards: NDArray[np.float64] = np.zeros(len(actions))
        terminated: bool = False
        truncated: bool = False
        self.step_count += 1
        for i in order:
            if actions[i] == self.actions.NORTH:
                next_pos = self.agents[i].north_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.EAST:
                next_pos = self.agents[i].east_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.SOUTH:
                next_pos = self.agents[i].south_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
            elif actions[i] == self.actions.WEST:
                next_pos = self.agents[i].west_pos()
                next_cell = self.grid.get(*next_pos)
                self.move_agent(rewards, i, next_cell, next_pos)
        if not self.respawn and self.collected_balls == self.num_balls:
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.grid.encode()
        return obs, rewards, terminated, truncated, self.info

    def phi_dim(self) -> int:
        """
        Helper method to get feature vector dimension

        Returns
        -------
        int
            length of feature vector = number of ball types
        """
        return self.num_ball_types
