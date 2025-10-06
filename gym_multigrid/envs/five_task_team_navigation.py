from typing import Literal, Optional, TypedDict, TypeAlias
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

from gym_multigrid.core.agent import NavigationActions, ActionsT, Agent
from gym_multigrid.core.agent import NAV_DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import AgentGoal, Wall, WorldObjT, Zone
from gym_multigrid.core.world import WorldT, TeamNavigationWorld
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing import Position
from gym_multigrid.core.object_group import ObjectGroup, EnvObjectGroup, ObjGroupT
from gym_multigrid.utils.subtasks import (
    PositionDist,
    SubtaskData,
    StateData,
    HLMDPConfig,
)


def get_initial_hlmdp_config(
    num_agents: int = 3,
    run_mode: Literal["manual"] | None = None,
    subtask_type: Literal["dependent", "independent"] = "dependent",
) -> HLMDPConfig:
    """gets the high level MDP configuration for this environment

    Parameters
    ----------
    num_agents : int
        number of agents in the enivronment

    Returns
    -------
    HLMDPConfig
        configuration for the high level MDP
    """

    if subtask_type == "dependent":
        termination_condition = "reach_goal_state_set"
    else:
        # independent subtasks
        termination_condition = "reach_assigned_goal_state"

    match num_agents:
        case 3:
            # state data
            if run_mode == "manual" or subtask_type == "independent":
                # have a pre-defined, deterministic ISD for each subtask
                state_data_tuple = (
                    StateData(
                        idx=0,
                        outgoing_init_state_dist=PositionDist(
                            states=[((1, 6), (1, 7), (1, 8))], probs=(1.0,)
                        ),
                    ),
                    StateData(
                        idx=1,
                        outgoing_init_state_dist=PositionDist(
                            states=[((6, 2), (6, 3), (6, 4))], probs=(1.0,)
                        ),
                    ),
                    StateData(
                        idx=2,
                        outgoing_init_state_dist=PositionDist(
                            states=[((9, 6), (9, 7), (9, 8))], probs=(1.0,)
                        ),
                    ),
                    StateData(
                        idx=3,
                        outgoing_init_state_dist=PositionDist(
                            states=[((6, 10), (6, 11), (6, 12))], probs=(1.0,)
                        ),
                    ),
                )

            else:
                state_data_tuple = (
                    StateData(
                        idx=0,
                        outgoing_init_state_dist=PositionDist(
                            states=[((1, 6), (1, 7), (1, 8))], probs=(1.0,)
                        ),
                    ),
                    StateData(
                        idx=1,
                        outgoing_init_state_dist=PositionDist(states=(), probs=()),
                    ),
                    StateData(
                        idx=2,
                        outgoing_init_state_dist=PositionDist(states=(), probs=()),
                    ),
                    StateData(
                        idx=3,
                        outgoing_init_state_dist=PositionDist(states=(), probs=()),
                    ),
                )

            # subtask data
            subtask_data_tuple = (
                SubtaskData(
                    edge=(0, 1),
                    idx=0,
                    termination_condition=termination_condition,
                    goal_state_set=((6, 2), (6, 3), (6, 4)),
                ),
                SubtaskData(
                    edge=(1, 2),
                    idx=1,
                    termination_condition=termination_condition,
                    goal_state_set=((9, 6), (9, 7), (9, 8)),
                ),
                SubtaskData(
                    edge=(0, 3),
                    idx=2,
                    termination_condition=termination_condition,
                    goal_state_set=((6, 10), (6, 11), (6, 12)),
                ),
                SubtaskData(
                    edge=(3, 2),
                    idx=3,
                    termination_condition=termination_condition,
                    goal_state_set=((9, 6), (9, 7), (9, 8)),
                ),
                SubtaskData(
                    edge=(2, 4),
                    idx=4,
                    termination_condition=termination_condition,
                    goal_state_set=((12, 6), (12, 7), (12, 8)),
                ),
            )

        case _:
            raise ValueError(
                "Chosen number of agents not implemented in the environment."
            )

    hlmdp_config: HLMDPConfig = HLMDPConfig(
        state_data_tuple=state_data_tuple, subtask_data_tuple=subtask_data_tuple
    )

    return hlmdp_config


# Classes used to interface with the env
class EnvInfo(TypedDict):
    """info about the environment used for PyMARL training"""

    state_shape: int
    obs_shape: int
    n_actions: int
    n_agents: int


class StepInfo(TypedDict):
    """info used in the env's step function"""

    success: bool


@dataclass
class RewardConfig:
    movement_reward: float
    agent_reach_goal_reward: float
    agent_leave_goal_reward: float
    all_agents_at_goal_reward: float


# reward_config = RewardConfig(
#     movement_reward=0.0,
#     agent_reach_goal_reward=0.9,
#     agent_leave_goal_reward=-1.0,
#     all_agents_at_goal_reward=1.0,
# )


reward_config = RewardConfig(
    movement_reward=0.0,
    agent_reach_goal_reward=0.4,
    agent_leave_goal_reward=-0.5,
    all_agents_at_goal_reward=1.0,
)


Observation: TypeAlias = (
    dict[str, NDArray[np.int_]] | NDArray[np.int_] | NDArray[np.float32]
)


class Detector:
    """detects agents in an area with a given object type"""

    def __init__(
        self,
        obj_type: str,
        group_index: int,
        visual_detect_prob: float,
        radio_detect_prob: float,
    ) -> None:
        """
        Parameters
        ----------
        obj_type : str
            object type that this detector will observe
        group_index : int
            index of group of objects that this detector will observe
        visual_detect_prob : float
            probability of detector observing an agent in a space with (obj_type, group_index) during a single time period (t, t+1)
        radio_detect_prob : float
            probability of detector observing an agent in a space with (obj_type, group_index) if that agent's communication value is >0
        """
        self.obj_type: str = obj_type
        self.group_index: int = group_index
        self.visual_detect_prob: float = visual_detect_prob
        self.radio_detect_prob: float = radio_detect_prob

    def detect_agents(
        self,
        agents: list[Agent],
        comms_val: float,
        obj_group_dict: dict[str, dict[int, ObjGroupT]],
        random_generator: np.random.Generator,
    ) -> bool:
        """get the stochastic observation of the detector

        Parameters
        ----------
        agents : list[Agent]
            list of all agents
        comms_val : float
            amount of communication used by the team of agents
        obj_group_dict : dict[str, dict[int, ObjGroupT]]
            group of objects in the environment
        random_generator : np.random.Generator
            random number generator

        Returns
        -------
        bool
            True if the detector observes an agent, False otherwise
        """
        obj_group: ObjGroupT = obj_group_dict[self.obj_type][self.group_index]
        for agent in agents:
            if self.detect_agent(agent, comms_val, obj_group, random_generator):
                return True
            else:
                pass

        return False

    def detect_agent(
        self,
        agent: Agent,
        comms_val: float,
        obj_group: ObjGroupT,
        random_generator: np.random.Generator,
    ) -> bool:
        if (agent.pos[0], agent.pos[1]) in obj_group.pos:
            visual_detect: bool = random_generator.uniform() < self.visual_detect_prob

            radio_detect: bool
            if comms_val > 0:
                radio_detect = random_generator.uniform() < self.radio_detect_prob
            else:
                radio_detect = False

            return visual_detect or radio_detect
        else:
            return False


class FiveTaskTeamNavigationEnv(MultiGridEnv):
    """team navigation environment"""

    metadata = {"render_fps": 10, "render_modes": ["human", "rgb_array"]}

    # setup and env properties
    def __init__(
        self,
        init_state_dist: PositionDist,
        height: int = 15,
        width: int = 14,
        num_agents: int = 2,
        p_intended_movement: float = 1.0,
        p_detect_visual: float = 0.005,
        zone_width: int = 1,
        comms_val: float = 1.0,
        actions_set: type[ActionsT] = NavigationActions,
        subtask_idx: int = 0,
        world: WorldT = TeamNavigationWorld,
        observation_option: Literal["goal", "all_goals"] = "all_goals",
        obs_type: Literal["array", "array_scaled"] = "array_scaled",
        reward_config: RewardConfig = reward_config,
        agent_dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Constructor for the TeamNavigationEnv class.

        Parameters
        ----------
        height : int = 9
            Height of the grid.
        width : int = 10
            Width of the grid.
        num_agents : int = 2
            number of agents in the environment.
        p_intended_movement : float = 0.95
            Probability of the intended movement.
            Should be in the range [0, 1].
        comms_val: float = 1.0
            Amount of communication used by the team of agents.
            Should be in the range [0, 1].
        subtask_idx: int = 0
            The current subtask index.
        observation_option : Literal["goal"] = "goal"
            Observation option.
            - "goal": The observation includes agent positions and position of the assigned goal.
        actions_set : type[ActionsT] = NavigationActions
            Set of actions for the agents.
            By default, there are five actions: "stay", "up", "right", "down", and "left".
        agent_dir_to_vec : list[NDArray[np.int_]] = NAV_DIR_TO_VEC
            Direction vectors for the agents.
            The length of the list should be equal to the number of actions in the actions set.
        reward_config: RewardConfig = reward_config
            Configuration for conditions that cause the reward function to output non-zero reward
        world : WorldT = LabyrinthWorld
            World for the environment.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Render mode for the environment.
        """
        # Do not read the init state dists from HLMDP config b/c
        # the distribution changes during the CM training algorithm.
        # init_state_dist has to be an arg pass into this env's init method
        self.init_state_dist: PositionDist = init_state_dist
        self.num_agents: int = num_agents
        self.p_intended_movement: float = p_intended_movement
        self.comms_val: float = comms_val
        self.subtask_idx: int = subtask_idx
        self.reward_config: RewardConfig = reward_config
        self.p_detect_visual = p_detect_visual
        self.zone_width = int(zone_width)

        # get the hlmdp data that doesn't change during the CM training algorithm
        self.hlmdp_config: HLMDPConfig = get_initial_hlmdp_config()
        subtask_data = self.hlmdp_config.subtask_data[subtask_idx]
        self.termination_condition = subtask_data.termination_condition
        self.goal_state_set = subtask_data.goal_state_set

        # observation config
        self.observation_option = observation_option
        self.obs_type = obs_type

        # agent config
        agent_view_size: int | None = None
        agents: list[Agent] = [
            Agent(
                world=world,
                index=i,
                view_size=agent_view_size,
                actions=actions_set,
                dir_to_vec=agent_dir_to_vec,
            )
            for i in range(num_agents)
        ]
        uncached_object_types: list[str] = ["agent"]

        # have to init this after the set of agents is constructed
        self.agent_goal_state_sets: dict[int, NDArray]
        self.build_goal_state_sets(agents)

        # basic grid init
        self.detectors: list[Detector]
        self.object_options: dict[str, WorldObjT] = {
            "goal": AgentGoal,
            "zone": Zone,
            "wall": Wall,
        }
        self.obj_group_dict: dict[str, dict[int, ObjGroupT]]
        self.init_grid: Grid
        # subtract 2 here b/c we assume an outer wall around the env that don't contribute to the height + width of the environment the agents can access
        self.obs_scaling: dict[Literal["x", "y", "obj_encoding"], np.int_] = {
            "x": width - 2,
            "y": height - 2,
            "obj_encoding": max(world.OBJECT_TO_IDX.values()),
        }

        super().__init__(
            agents=agents,
            width=width,
            height=height,
            actions_set=actions_set,
            world=world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )

        # define the action space
        self.action_space = spaces.MultiDiscrete(
            [len(self.actions) for _ in range(self.num_agents)]
        )

    def _set_observation_space(self) -> spaces.Box:
        max_x: int = self.width - 1
        max_y: int = self.height - 1

        obs_shape = self.reset()[0].shape

        if self.obs_type in ["array", "array_scaled"]:
            if self.obs_type == "array":
                max_val = np.max((max_x, max_y))
            else:
                max_val = 1

            observation_space = spaces.Box(
                low=np.zeros(obs_shape),
                high=max_val * np.ones(obs_shape),
                dtype=np.float32,
            )

        else:
            raise ValueError(f"Invalid observation option: {self.observation_option}")

        return observation_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[NDArray[np.int_], StepInfo]:

        super().reset(seed=seed, options=options)

        obs: Observation = self.get_obs()
        info: StepInfo = self._get_step_info()

        return obs, info

    def _gen_grid(self, width, height) -> None:
        self.grid = Grid(width, height, self.world)

        # make a list of all the objects you want to spawn in the env
        env_object_config = [
            # surrounding wall
            EnvObjectGroup(
                obj_type="wall",
                group_index=0,
                pos=((0, 0, self.width, self.height),),
                color="grey",
                spawned_subtask_indices=(0, 1, 2, 3, 4),
                fill_mode="empty",
            ),
            # middle wall
            EnvObjectGroup(
                obj_type="wall",
                group_index=1,
                pos=((4, 7, 3, 1),),
                color="grey",
                spawned_subtask_indices=(0, 1, 2, 3, 4),
                fill_mode="empty",
            ),
            # blue zone
            EnvObjectGroup(
                obj_type="zone",
                group_index=0,
                pos=((4, 1, self.zone_width, 6),),
                color="blue",
                spawned_subtask_indices=(0, 1, 2, 3, 4),
                fill_mode="empty",
            ),
            # red zone
            EnvObjectGroup(
                obj_type="zone",
                group_index=1,
                pos=((4, 8, self.zone_width, 6),),
                color="red",
                spawned_subtask_indices=(0, 1, 2, 3, 4),
                fill_mode="empty",
            ),
        ]

        # initialize the detectors
        self.detectors: list[Detector] = [
            Detector(
                obj_type="zone",
                group_index=0,
                visual_detect_prob=self.p_detect_visual,
                radio_detect_prob=0.0,
            ),
            Detector(
                obj_type="zone",
                group_index=1,
                visual_detect_prob=self.p_detect_visual,
                radio_detect_prob=0.1,
            ),
        ]

        # place the goals for the current subtask
        for subtask_idx, data in self.hlmdp_config.subtask_data.items():
            if subtask_idx == self.subtask_idx:
                env_object_config.append(
                    EnvObjectGroup(
                        obj_type="goal",
                        group_index=0,
                        pos=self.goal_state_set,
                        color="green",
                        spawned_subtask_indices=(subtask_idx,),
                        fill_mode="empty",
                    )
                )

        obj_group_dict: dict[str, dict[int, ObjGroupT]] = {}

        # Place objects
        for obj_group_config in env_object_config:
            obj_type: str = obj_group_config.obj_type
            group_index: int = obj_group_config.group_index

            if obj_type not in obj_group_dict:
                obj_group_dict[obj_type] = {}
            else:
                pass

            obj_group_dict[obj_type][group_index] = ObjectGroup(
                object_options=self.object_options, **asdict(obj_group_config)
            )
            obj_group_dict[obj_type][group_index].put_objects(self.grid, self.world)

        self.obj_group_dict = obj_group_dict

        # init_grid has to be defined before the agents are placed
        self.init_grid: Grid = self.grid.copy()

        # Place the agents by sampling a state from the initial state distribution
        init_pos = self.np_random.choice(
            self.init_state_dist.states, p=self.init_state_dist.probs
        )

        assert len(self.agents) == len(init_pos)
        for agent, pos in zip(self.agents, init_pos):
            self.place_agent(agent, pos)

    def get_obs(self) -> Observation:
        # get empty map of the env
        map_obs: NDArray[np.int_] = self._get_map_obs()
        goal_obs: NDArray[np.int_] = self._get_goal_obs()

        # get the agent's x-y coordinates
        for agent_idx, agent in enumerate(self.agents):
            agent_obs = (agent.pos[0], agent.pos[1], self.world.OBJECT_TO_IDX["agent"])

            if self.obs_type == "array_scaled":
                agent_obs = np.array(
                    (
                        agent_obs[0] / self.obs_scaling["x"],
                        agent_obs[1] / self.obs_scaling["y"],
                        agent_obs[2] / self.obs_scaling["obj_encoding"],
                    ),
                    dtype=np.float32,
                )

            if self.obs_type in ["array", "array_scaled"]:
                agent_obs = np.concat(
                    (agent_obs, goal_obs.flatten(), map_obs.flatten())
                )

                # set the size of the team-level obs
                if agent_idx == 0:
                    obs: NDArray[np.int_] = np.zeros(
                        (self.num_agents, len(agent_obs)), dtype=np.float32
                    )

                obs[agent.index, :] = agent_obs.flatten()

        return obs

    def _get_goal_obs(self) -> NDArray[np.int_]:
        """get agent observation of goal objects

        Returns
        -------
        NDArray[np.int_]
            a 2D array of goal observations
        """
        goal_obs: NDArray[np.int_]

        if self.observation_option == "all_goal_states_all_subtasks":
            # each agent will have the same obs of the goal states
            # they can see all the goal states of all the subtasks

            # set of goal states across all subtasks
            goal_state_set_all_subtasks: list[tuple[float, float]] = []

            for _, data in self.hlmdp_config.subtask_data.items():
                goal_state_set_single_subtask = data.goal_state_set

                for goal_state in goal_state_set_single_subtask:
                    goal_obs_single = (
                        goal_state[0],
                        goal_state[1],
                        self.world.OBJECT_TO_IDX["goal"],
                    )

                    # scale the observation to have entries in [0, 1]
                    if self.obs_type == "array_scaled":
                        goal_obs_single = (
                            goal_obs_single[0] / self.obs_scaling["x"],
                            goal_obs_single[1] / self.obs_scaling["y"],
                            goal_obs_single[2] / self.obs_scaling["obj_encoding"],
                        )

                    goal_state_set_all_subtasks.append(goal_obs_single)

        goal_obs = np.array(goal_state_set_all_subtasks)
        return goal_obs

    def _get_map_obs(self) -> NDArray[np.int_]:

        # I want a (max_width, max_height, encode_dim) size np array that represents the map without any agents or goals in it
        ## If the env doesn't change, I just need to run this function once at the start of the episode
        ## eh, just run it each step in get_obs

        # loop over every x-y position, get the object type, and get the encoding idx of that object type
        # this will start out as a (width, height, 3) tensor, but it will get flattened, which will remove the fact that the tensor directly represents the env
        ## so we need the x-y position data to make sure that information is not lost
        if self.obs_type in ["array", "array_scaled"]:
            env_map: NDArray[np.int_] = np.zeros(
                (self.height, self.width, 3), dtype=np.int_
            )

            for x in range(self.width):
                for y in range(self.height):
                    obj = self.grid.get(x, y)

                    # we do not include information about the positions of other agents or the goals in the map view
                    if (obj is None) or obj.type in ["goal", "agent"]:
                        obj_encoding: int = self.world.OBJECT_TO_IDX["empty"]
                    else:
                        obj_encoding: int = self.world.OBJECT_TO_IDX[obj.type]

                    if self.obs_type == "array_scaled":
                        env_map[y, x, :] = np.array(
                            [
                                x / self.obs_scaling["x"],
                                y / self.obs_scaling["y"],
                                obj_encoding / self.obs_scaling["obj_encoding"],
                            ]
                        )

                    else:
                        env_map[y, x, :] = np.array([x, y, obj_encoding])

        else:
            raise NotImplementedError

        return env_map

    def get_env_info(self) -> EnvInfo:
        # obs_shape should only be the shape of a single agent
        # self.observation_space.shape = (n_agents, dim_1_size, dim_2_size, ...)
        obs_shape = int(np.prod(self.observation_space.shape[1:]))

        env_info: EnvInfo = {
            "state_shape": self.width * self.height * self.world.encode_dim,
            "obs_shape": obs_shape,
            "n_actions": len(self.actions),
            "n_agents": self.num_agents,
        }

        return env_info

    def get_avail_actions(self, verbose=False) -> list[list[bool]]:
        """gets available actions for each agent

        Returns
        -------
        avail_actions: list[list[bool]]
            available actions for each agent
        """
        avail_actions = []
        for agent in self.agents:
            avail_actions.append(self._get_avail_actions_agent(agent))

        if verbose:
            self.print_avail_actions_str(avail_actions)

        return avail_actions

    def print_avail_actions_str(self, avail_actions):
        if self.actions == NavigationActions:
            actions = [a.name for a in NavigationActions]

            print("Available actions")
            for agent_idx, avail_actions_agent in enumerate(avail_actions):
                action_str = ""
                for i, avail in enumerate(avail_actions_agent):
                    if avail:
                        action_str += f"{actions[i]},  "

                print(f"Agent: {agent_idx} --- {action_str}")

    def _get_avail_actions_agent(self, agent: Agent) -> list[bool]:
        if self.actions == NavigationActions:
            avail_actions_tmp: dict[int, bool] = {}

            # populate with filler data
            for action in NavigationActions:
                avail_actions_tmp[action.value] = True

            # set the values in avail_actions
            # you should be able to set the desired order of the neighbor positions here
            neighbor_positions: dict[str, NDArray[np.int_]] = (
                agent.get_all_neighbor_pos()
            )

            for direction, pos in neighbor_positions.items():
                neighbor_cell = self.grid.get(*pos)
                if (neighbor_cell is None) or (neighbor_cell.can_overlap()):
                    continue
                else:
                    avail_actions_tmp[NavigationActions[direction].value] = False

            # turn avail_actions into a list with the ordering of the actions same as in NavigationActions
            avail_actions: list[bool] = list(avail_actions_tmp.values())

        else:
            raise NotImplementedError(
                "This action space is not implemented in _get_avail_actions"
            )

        return avail_actions

    # general env step logic
    def step(
        self,
        actions: NDArray[np.int_],
    ) -> tuple[Observation, float, bool, bool, StepInfo]:

        self.step_count += 1

        # transition from s_t to s_{t+1}
        curr_state, next_state = self._move_agents(actions=actions)

        # get reward for transition (s_t, a_t, s_{t+1})
        reward: float = self._reward(
            curr_state=curr_state, actions=actions, next_state=next_state
        )

        # get observation from being in s_{t+1}
        obs: Observation = self.get_obs()

        terminated, all_at_goal = self._terminated(next_state=next_state)

        # truncated is handled by a Gymnasium wrapper
        truncated: bool = False

        info: StepInfo = self._get_step_info(all_at_goal)

        return obs, reward, terminated, truncated, info

    # agent movement
    def _move_agents(
        self, actions: list[int]
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Move agents based on the chosen action and environment randomness

        Parameters
        ----------
        actions : list[int]
            Actions to take.

        Returns
        -------
        curr_state : NDArray[np.int_]
            (x, y) positions of the agents at time t (before the state transition).
        next_state : NDArray[np.int_]
            (x, y) positions of the agents at time t+1 (after the state transition).
        """
        # Randomly generate the order of the agents
        agent_indices: list[int] = list(range(self.num_agents))
        self.np_random.shuffle(agent_indices)

        # placeholder for the (x, y) positions of all the agents
        ## assumes no other state information matters for the reward calculation
        curr_state: NDArray[np.int_] = np.zeros((self.num_agents, 2))
        next_state: NDArray[np.int_] = np.zeros((self.num_agents, 2))

        for i in agent_indices:
            curr_state[i, :] = self.agents[i].pos
            self._move_agent(action=actions[i], agent=self.agents[i])

            # set next_state_list[:, i] to be the next (x, y) position of agent i
            next_state[i, :] = self.agents[i].pos

        return curr_state, next_state

    def _move_agent(self, action: int, agent: Agent) -> None:
        """
        Move agents based on the action.

        Parameters
        ----------
        action : int
            Action chosen by the agent.
        agent : Agent
            Agent to move.
        """
        ################
        # pick out the next position the agent would go to based on its action if there were no randomness in the env's transition function
        ## this logic would not be necessary if there were action masking based on avail_actions that would prevent an agent from selecting an action like "walk into the wall", but I guess that isn't standard in gymnasium envs
        chosen_next_state: Position
        assert agent.pos is not None
        chosen_next_state = agent.pos + agent.dir_to_vec[action]

        # define a flag to tell if the agent will move or not
        chosen_next_state_in_available_pos: bool = False
        available_pos: list[Position] = self._get_available_pos(agent=agent)
        for pos in available_pos:
            if np.array_equal(pos, chosen_next_state):
                chosen_next_state_in_available_pos = True
                break
            else:
                pass

        ################
        # define agent movement logic
        # agent moves to a new state
        if chosen_next_state_in_available_pos:
            action_probs: list[float] = []

            # assign probabilities to each possible action
            # assigns equal probability to all other available next positions
            # EX: if the agent chose "up", and P(s, "up", s_up) = 0.97, then
            # P(s, "right", s_right) = P(s, "down", s_down) = P(s, "left", s_left) =
            # (1 - 0.97) / 3 = 0.1
            for pos in available_pos:
                action_probs.append(
                    self.p_intended_movement
                    if np.array_equal(pos, chosen_next_state)
                    else (1 - self.p_intended_movement) / (len(available_pos) - 1)
                )

            # normalize the action probabilities
            action_probs = np.array(action_probs) / np.sum(action_probs)
            avail_pos_indices: list[int] = list(range(len(available_pos)))

            # chose the next position based on the randomly chosen action
            next_state_idx = self.np_random.choice(avail_pos_indices, p=action_probs)
            next_state = available_pos[next_state_idx]

            # make the agent move
            next_cell: WorldObjT | None = self.grid.get(*next_state)
            if next_cell is None:
                agent.move(next_state, self.grid, self.init_grid, bg_color=None)
            elif next_cell.can_overlap():
                agent.move(
                    next_state, self.grid, self.init_grid, bg_color=next_cell.bg_color
                )
            else:
                raise ValueError(
                    f"Invalid action f{action} and position f{next_state} for agent {agent.index}. \
                        Available positions: {available_pos}"
                )

        # agent stays in its current state
        else:
            pass

    def _get_available_pos(self, agent: Agent) -> list[Position]:
        possible_pos: list[Position] = []

        for direction in agent.dir_to_vec:
            next_state: Position = agent.pos + direction

            if (
                next_state[0] < 0
                or next_state[1] < 0
                or next_state[0] >= self.width
                or next_state[1] >= self.height
            ):
                pass
            else:
                next_cell: WorldObjT | None = self.grid.get(*next_state)

                if self.grid.get(*next_state) is None:
                    possible_pos.append(next_state)
                elif next_cell.can_overlap():
                    possible_pos.append(next_state)
                else:
                    pass

        return possible_pos

    # reward function
    def _reward(
        self,
        curr_state: NDArray[np.int_],
        actions: NDArray[np.int_],
        next_state: NDArray[np.int_],
    ) -> float:
        """
        Parameters
        ----------
        curr_state : NDArray[np.int_]
            current joint state
        actions : NDArray[np.int_]
            current agent actions
        next_state : NDArray[np.int_]
            next joint state
        Returns
        -------
        float
            reward
        """
        reward: float = 0.0

        reward += self._reward_movement(actions)
        reward += self._reward_reach_goal(curr_state, next_state)
        reward += self._reward_leave_goal(curr_state, next_state)
        reward += self._reward_all_at_goal(next_state)

        return reward

    def _reward_movement(self, actions: NDArray[np.int_]) -> float:
        """Cost incurred by each action that is not "stay"
        Parameters
        ----------
        actions : NDArray[np.int_]
            agent actions

        Returns
        -------
        float
            reward
        """

        reward = 0.0
        reward += self.reward_config.movement_reward * np.sum(
            actions != self.actions.stay
        )

        return reward

    def _reward_reach_goal(
        self,
        curr_state: NDArray[np.int_],
        next_state: NDArray[np.int_],
    ) -> float:
        """Reward for an agent reaching its assigned final goal state

        Parameters
        ----------
        curr_state : NDArray[np.int_]
            current state
        next_state : NDArray[np.int_]
            next state
        Returns
        -------
        float
            reward
        """
        reward: float = 0.0
        n_agents_reach_goal: int = 0

        for agent in self.agents:
            # agent is not currently in a goal state
            condition_1 = not np.any(
                np.all(
                    curr_state[agent.index, :]
                    == self.agent_goal_state_sets[agent.index],
                    axis=1,
                )
            )

            # agent's next state is one of its valid goal states
            condition_2 = np.any(
                np.all(
                    next_state[agent.index, :]
                    == self.agent_goal_state_sets[agent.index],
                    axis=1,
                )
            )


            if condition_1 and condition_2:
                # print(f"Agent {agent.index} reached its goal state")
                n_agents_reach_goal += 1

        reward += n_agents_reach_goal * self.reward_config.agent_reach_goal_reward

        return reward

    def _reward_leave_goal(
        self,
        curr_state: NDArray[np.int_],
        next_state: NDArray[np.int_],
    ) -> float:
        """Reward for an agent leaving its assigned final goal state

        Parameters
        ----------
        curr_state : NDArray[np.int_]
            current state
        next_state : NDArray[np.int_]
            next state

        Returns
        -------
        float
            reward
        """
        reward: float = 0
        n_agents_leave_goal: int = 0

        for agent in self.agents:
            # agent's current state is a valid goal state
            condition_1 = np.any(
                np.all(
                    curr_state[agent.index, :]
                    == self.agent_goal_state_sets[agent.index],
                    axis=1,
                )
            )

            # # agent's next state is not a valid goal state
            # condition_2 = not np.any(
            #     np.all(
            #         next_state[agent.index, :]
            #         == self.agent_goal_state_sets[agent.index],
            #         axis=1,
            #     )
            # )

            # # agent's next state is different from its current state
            condition_2 = not np.array_equal(curr_state[agent.index, :], next_state[agent.index, :])


            if condition_1 and condition_2:
                # print(f"Agent {agent.index} left a goal state")
                n_agents_leave_goal += 1

        reward += n_agents_leave_goal * self.reward_config.agent_leave_goal_reward

        return reward

    def _reward_all_at_goal(self, next_state: NDArray[np.int_]) -> float:
        """Reward the team for all being at their final assigned states

        Parameters
        ----------
        next_state : NDArray[np.int_]
            next state

        Returns
        -------
        float
            reward
        """
        reward = 0.0
        all_at_goal_flag = self._check_all_agents_reached_goal(next_state=next_state)

        reward += all_at_goal_flag * self.reward_config.all_agents_at_goal_reward

        return reward

    # termination function
    def _terminated(self, next_state: NDArray[np.int_]) -> tuple[bool, bool]:
        """check terminated conditions

        Parameters
        ----------
        next_state : NDArray[np.int_]
            next state

        Returns
        -------
        tuple[bool, bool]
            terminated status, flag that's true when all agents are at a goal state
        """
        # I define two different variables here b/c we may consider other conditions
        # in defining "terminated" in the future or in other envs
        terminated = (self._check_all_agents_reached_goal(next_state=next_state)) or (
            self._check_any_agent_detected()
        )

        all_at_goal_flag = self._check_all_agents_reached_goal(next_state=next_state)

        return terminated, all_at_goal_flag

    def _check_all_agents_reached_goal(self, next_state: NDArray[np.int_]) -> bool:
        """check if each agent is in a valid goal state"""

        reached_goal: NDArray = np.zeros(len(self.agents))

        for i, agent in enumerate(self.agents):
            reached_goal[i] = np.any(
                np.all(
                    next_state[agent.index, :]
                    == self.agent_goal_state_sets[agent.index],
                    axis=1,
                )
            )

        # Convert from np bool to python bool b/c
        # np bools cannot be interpreted as integers.
        # This is an issue that comes up later in the training pipeline.
        all_at_goal_flag = bool(np.all(reached_goal))

        return all_at_goal_flag

    def _check_any_agent_detected(self) -> bool:
        """check whether any agent was detected by the detection zones in the environment

        Returns
        -------
        bool
            True if any agent was detected
        """
        detected: bool = False
        for i, detector in enumerate(self.detectors):
            if detector.detect_agents(
                self.agents, self.comms_val, self.obj_group_dict, self.np_random
            ):
                detected = True
                break

        return detected

    # step info
    def _get_step_info(self, all_at_goal: bool = False) -> StepInfo:
        """other env info to be returned by the env's step method"""
        step_info: StepInfo = {"success": all_at_goal}

        return step_info
