from typing import Literal, Optional, TypedDict, TypeAlias
from dataclasses import asdict, dataclass
import pdb
import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces

from gym_multigrid.core.agent import NavigationActions, ActionsT, TypedAgent
from gym_multigrid.core.constants import NAV_DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Wall, WorldObjT
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


def get_hlmdp_config(num_subtasks) -> HLMDPConfig:
    match num_subtasks:
        # The following parameters are irrelevant in this environment, so are filled with filler data
        # outgoing_init_state_dist
        # final_state
        # termination_condition

        # 2 subtasks
        case 2:
            state_data_tuple = (
                StateData(
                    idx=0,
                    outgoing_init_state_dist=PositionDist(
                        probs=(1.0,), states=((0, 0),)
                    ),
                ),
                StateData(
                    idx=1,
                    outgoing_init_state_dist=PositionDist(
                        probs=(1.0,), states=((0, 0),)
                    ),
                ),
            )

            subtask_data_tuple = (
                SubtaskData(
                    edge=(0, 1),
                    idx=0,
                    final_state=((0, 0),),
                    termination_condition="0",
                ),
                SubtaskData(
                    edge=(1, 2),
                    idx=1,
                    final_state=((0, 0),),
                    termination_condition="0",
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
    coordinated_joint_action: float


# Env config
reward_config = RewardConfig(coordinated_joint_action=1.0)


Observation: TypeAlias = (
    dict[str, NDArray[np.int_]] | NDArray[np.int_] | NDArray[np.float32]
)


class OneStepCoordinationNoGroupsEnv(MultiGridEnv):
    """team navigation environment"""

    metadata = {"render_fps": 10, "render_modes": ["human", "rgb_array"]}

    # setup and env properties
    def __init__(
        self,
        num_agents: int = 4,
        num_type_1_thres: int = 2,
        subtask_idx: int = 0,
        p_intended_movement: float = 1.0,
        actions_set: type[ActionsT] = NavigationActions,
        world: WorldT = TeamNavigationWorld,
        obs_type: Literal["array"] = "array",
        reward_config: RewardConfig = reward_config,
        agent_dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ) -> None:
        """
        Constructor for the TeamNavigationEnv class.

        Parameters
        ----------
        num_agents: int = 4
            Number of agents in the environment
        num_type_1_thres: int = 2
            Number of Type 1 agents that have to take the down action to receive the high reward
        p_intended_movement : float = 1.0
            Probability of the intended movement.
        actions_set : type[ActionsT] = NavigationActions
            Set of actions for the agents.
            By default, there are three actions: up, down, stay.
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
        self.num_agents = num_agents
        self.num_type_1_thres = num_type_1_thres
        self.subtask_idx = subtask_idx
        self.p_intended_movement: float = p_intended_movement
        self.reward_config: RewardConfig = reward_config

        width: int = 2 * self.num_agents + 1
        height: int = 4
        self.num_type_1_agents: int

        # observation config
        self.obs_type: Literal["array"] = obs_type

        # agent config
        agent_view_size: int | None = None
        agents: list[TypedAgent] = [
            TypedAgent(
                world=world,
                index=i,
                view_size=agent_view_size,
                actions=actions_set,
                dir_to_vec=agent_dir_to_vec,
            )
            for i in range(self.num_agents)
        ]
        uncached_object_types: list[str] = ["agent"]
        self.agent_types: dict[int, list[int]]

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

        # basic grid init
        self.object_options: dict[str, WorldObjT] = {
            "wall": Wall,
        }
        self.obj_group_dict: dict[str, dict[int, ObjGroupT]]
        self.init_grid: Grid

    def _assign_agent_types(self) -> None:
        """assigns agent types uniformly randomly"""
        agent_idxs: NDArray = np.array([agent.index for agent in self.agents])

        type_1_agent_idxs: NDArray = self.np_random.choice(
            agent_idxs,
            size=self.num_type_1_agents,
            replace=False,
        )

        for i in type_1_agent_idxs:
            agent = self.agents[i]
            agent.agent_type = 1

    def _set_observation_space(self) -> spaces.Box:
        # num agents + num of agent types
        obs_size = (self.num_agents, (self.num_agents + 1) + 2)
        observation_space = spaces.Box(
            low=np.zeros(obs_size),
            high=np.ones(obs_size),
            dtype=np.float32,
        )

        return observation_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[Observation, StepInfo]:

        # reset agent types for the next episode
        for agent in self.agents:
            agent.agent_type = 0

        super().reset(seed=seed, options=options)

        obs = self.get_obs()
        info = self._get_step_info()

        return obs, info

    def _gen_grid(self, width, height) -> None:
        self.grid = Grid(width, height, self.world)

        self.num_type_1_agents = self.np_random.choice(
            np.arange(1, self.num_agents + 1)
        )

        # assign agents to types
        self._assign_agent_types()

        # make a list of all the objects you want to spawn in the env
        env_object_config = []

        # surrounding wall
        env_object_config.append(
            EnvObjectGroup(
                obj_type="wall",
                group_index=0,
                pos=((0, 0, self.width, self.height),),
                color="grey",
                fill_mode="empty",
            ),
        )

        # vertical walls dividing agents
        for i in range(self.num_agents):
            x = 2 * i

            env_object_config.append(
                EnvObjectGroup(
                    obj_type="wall",
                    group_index=0,
                    pos=((x, 0, 1, 3),),
                    color="grey",
                    fill_mode="empty",
                ),
            )

        # get agent positions
        for i in range(self.num_agents):
            x = 2 * i + 1
            y = 1
            self.agents[i].pos = (x, y)

        # place a wall below the agents that are type 0
        for agent in self.agents:
            if agent.agent_type == 0:
                x, y = agent.pos[0], agent.pos[1] + 1

                env_object_config.append(
                    EnvObjectGroup(
                        obj_type="wall",
                        group_index=0,
                        pos=((x, y, 1, 1),),
                        color="grey",
                        fill_mode="empty",
                    ),
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

        # Place the agents
        for i, agent in enumerate(self.agents):
            self.place_agent(agent, agent.pos)

    def get_obs(self) -> Observation:
        obs_list: list = []

        for agent_idx, agent in enumerate(self.agents):
            # one-hot representation of num_type_1_thres (part of the environment state)
            one_hot_thres = np.zeros(self.num_agents + 1)
            one_hot_thres[self.num_type_1_thres] = 1

            # one-hot representation of agent type
            one_hot_type = np.zeros(2)
            one_hot_type[agent.agent_type] = 1

            agent_obs = np.concatenate((one_hot_thres, one_hot_type))

            obs_list.append(agent_obs)

        # convert to np array of size N x F
        obs: NDArray = np.array(obs_list)
        return obs

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

    def _get_avail_actions_agent(self, agent: TypedAgent) -> list[bool]:
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
        self._move_agents(actions=actions)

        # get reward for transition (s_t, a_t, s_{t+1})
        reward = self._reward(actions=actions)

        # get observation from being in s_{t+1}
        obs = self.get_obs()

        terminated = self._terminated()

        # truncated is handled by a Gymnasium wrapper
        truncated: bool = False

        info = self._get_step_info(reward)

        return obs, reward, terminated, truncated, info

    # agent movement
    def _move_agents(
        self, actions: NDArray[np.int_]
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

    def _move_agent(self, action: int, agent: TypedAgent) -> None:
        """
        Move agents based on the action.

        Parameters
        ----------
        action : int
            Action chosen by the agent.
        agent : TypedAgent
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

    def _get_available_pos(self, agent: TypedAgent) -> list[Position]:
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
        actions: NDArray[np.int_],
    ) -> float:
        reward: float = 0.0

        num_down_actions = len(np.where(actions == self.actions.down)[0])

        reward_condition_0: bool = (
            self.num_type_1_agents >= self.num_type_1_thres
        ) and (num_down_actions == self.num_type_1_agents)

        reward_condition_1: bool = (
            self.num_type_1_agents < self.num_type_1_thres
        ) and (num_down_actions == 0)

        if reward_condition_0 or reward_condition_1:
            reward += reward_config.coordinated_joint_action
        else:
            reward += -1.0 * (num_down_actions / self.num_type_1_agents)

        return reward

    # termination function
    def _terminated(
        self,
    ) -> bool:
        """
        Returns
        -------
        bool
            terminated tells whether the env is terminated or not
        """

        return True

    # step info
    def _get_step_info(self, reward: float = 0.0) -> StepInfo:
        """get info to be returned in the step function"""
        if reward == self.reward_config.coordinated_joint_action:
            success = True
        else:
            success = False

        step_info: StepInfo = {"success": success}

        return step_info
