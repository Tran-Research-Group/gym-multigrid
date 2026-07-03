import itertools as it
import pandas as pd
from typing import Literal, Optional
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces, Env

from gym_multigrid.core.constants import COLORS


class MDPAgent:
    # simple agent class for a simple MDP
    def __init__(self, init_state: int) -> None:
        self._state = init_state
        self._prev_state = init_state

    def reset(self, init_state: int):
        self._state = init_state
        self._prev_state = init_state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def prev_state(self):
        return self._prev_state

    @prev_state.setter
    def prev_state(self, value):
        self._prev_state = value


class ProjectMDP(Env):
    # currently only hard coded for LBF env
    """Using terminology from the project scheduling literature, this MDP represents a project which consists of multiple tasks with ordering (precedence) constraints, pre-defined transitions, and state-dependent action spaces."""

    def __init__(
        self,
        num_rooms: int,
        msg_budget_per_agent: list[int],
        task_type: Literal["atomic", "composed"] = "composed",
    ):
        super().__init__()

        self.agent = MDPAgent(init_state=0)
        self.tasks: list[tuple]
        self.init_state: int
        self.goal_state: int
        self.fail_state: int
        self.state_space: NDArray[np.int_]
        self.successor_map: dict[tuple[int, tuple], int]
        self.msg_budget_per_agent = msg_budget_per_agent

        self._build_env(
            num_rooms=num_rooms,
            task_type=task_type,
        )

        self.task_completed: bool = False

        # stuff for MDP rendering
        self.graph: Optional[nx.Graph] = None
        # scale colors to be in [0, 1] for rendering
        self.colors = {k: v / 255 for k, v in COLORS.items()}
        self.node_colors: list
        self.edge_widths: dict = {
            "normal": 1,
            "highlight": 2.5,
        }

        # n_waypoints = n_agents is a mathematically different task from !=
        # you want to use a pandas df here for easier bookkeeping
        # self.observation_space: spaces.Discrete
        # self.action_space: spaces.Discrete

        # self.df_state: pd.DataFrame
        # df_task

    def _build_env(
        self,
        num_rooms: int,
        task_type: Literal["atomic", "composed"],
    ):
        """
        assume 1 set of waypoints per room
        set of atomic tasks that can advance the state in the MDP
         - clear a room of fruit
         - all agents reach a waypoint for the current room
         - composed task = "current room cleared of fruit" and "all agents reach the current room's waypoint" are True
        """

        match (num_rooms, task_type):
            case (2, "composed"):
                """
                0 -> 1 -> 2
                """
                # need to pre-define the tasks in the project MDP
                # tasks = edges in a graph
                self.tasks: list[tuple] = [(0, 1), (1, 2)]

            case _:
                raise NotImplementedError

        self.state_space = [i for i in range(0, len(self.tasks) + 2)]
        self.init_state = self.state_space[0]
        self.goal_state = self.state_space[-2]
        self.fail_state = self.state_space[-1]

        # include "stay" task for self-transition of absorbing states
        for state in [self.goal_state, self.fail_state]:
            self.tasks.append((state, state))

        self.observation_space = spaces.Discrete(n=len(self.state_space))

        # Action space: 2D discrete-continuous vector (task_idx, comms_val in [0, 1])
        self.n_tasks = len(self.tasks)
        self.action_space = spaces.Box(
            low=np.array([0, 0.0]),
            high=np.array([self.n_tasks - 1, 1.0]),
            dtype=np.float32,
        )

        # transition probs
        self._transition_probs: pd.DataFrame
        transition_probs: list[dict] = []

        # init probs as None until we have real data
        self.successor_map = {}
        for edge in self.tasks:
            curr_state, chosen_next_state = edge
            if curr_state == chosen_next_state:
                # add dummy actions for self-transition for goal state and fail state
                # dummy action for absorbing states always has a comms val of 0 since it isn't a real task
                action = (chosen_next_state, 0.0)

                self.successor_map[(curr_state, action)] = chosen_next_state
                transition_probs.append(
                    {
                        "state": curr_state,
                        "state_type": self._get_state_type(curr_state),
                        "action": action,
                        "next_state": chosen_next_state,
                        "prob": 1.0,
                    }
                )

            else:
                for budget in self.msg_budget_per_agent:
                    action = (chosen_next_state, budget)

                    next_states = [chosen_next_state, self.fail_state]
                    for next_state in next_states:
                        self.successor_map[(curr_state, action)] = chosen_next_state
                        transition_probs.append(
                            {
                                "state": curr_state,
                                "state_type": self._get_state_type(curr_state),
                                "action": action,
                                "next_state": next_state,
                                "prob": None,
                            }
                        )

        self._transition_probs = pd.DataFrame.from_records(transition_probs)

    def _get_state_type(self, state: int) -> Literal["goal", "fail", "normal"]:
        match state:
            case self.goal_state:
                state_type = "goal"
            case self.fail_state:
                state_type = "fail"
            case _:
                state_type = "normal"
        return state_type

    def step(
        self,
        action: dict,
        task_completed: bool,
        project_failed: bool,
    ) -> tuple[int, float, bool, bool, dict]:

        # Due to how time steps work in the PYMARL runner, need to set task_completed
        # so it is seen in the state getter to populate pre_transition_data to be used to select actions
        self.task_completed = False

        if project_failed:
            next_state = self.fail_state

        elif task_completed:
            self.task_completed = True
            action_tuple = self._get_action_tuple(action)
            # take action given by the hl agent
            next_state = self.successor_map[(self.agent.state, action_tuple)]

        else:
            # task still in progress
            next_state = self.agent.state

        # update agent state
        self.agent.prev_state = self.agent.state
        self.agent.state = next_state

        # Determine reward and termination
        terminated = self.agent.state == self.goal_state
        project_failed = self.agent.state == self.fail_state

        # reward = 1.0 if terminated else (-0.01 if failed else 0.0)
        reward = 0.0

        # truncated handled by TimeLimit wrapper on this MDP
        # or the low-level env in a hierarchical setup
        truncated = False

        obs = self.state
        env_info: dict = {"project_failed": project_failed}

        return (
            obs,
            reward,
            terminated,
            truncated,
            env_info,
        )

    def _get_action_tuple(self, action: dict) -> tuple:
        chosen_next_state = action["chosen_next_state"]
        msg_budget_raw = action["comms_budget"]

        # Discretize comms value to nearest level
        # only used if sampling from the action space for development purposes
        message_budget: float = self.msg_budget_per_agent[
            np.argmin(np.abs(np.array(self.msg_budget_per_agent) - msg_budget_raw))
        ]

        action_tuple = (chosen_next_state, message_budget)
        return action_tuple

    def reset(
        self, seed: Optional[int] = None, options: dict = None
    ) -> tuple[NDArray, dict]:
        """
        Reset MDP to initial state following Gymnasium API.

        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility
        options : dict, optional
            Additional options for starting state, e.g. {'hl_start_state': 1}

        Returns
        -------
        tuple[int, dict]
            (observation, info) where observation is the initial MDP state
        """
        super().reset(seed=seed)
        start_state = self.init_state
        if options is not None and "hl_start_state" in options:
            start_state = int(options["hl_start_state"])

        self.agent.reset(start_state)
        self.task_completed = False

        obs: NDArray = self.state
        info: dict = {"hl_start_state": self.agent.state}

        return obs, info

    @property
    def state(self) -> NDArray:
        return np.array([self.agent.state, self.task_completed])

    @property
    def transition_probs(self):
        return self._transition_probs

    @transition_probs.setter
    def transition_probs(self, df_data: pd.DataFrame):
        # the trans agenda is here bwahaha >:D
        df_trans = self._transition_probs

        for _, row in df_data.iterrows():
            # task success rate
            df_trans.loc[
                (df_trans.state == row.hl_start_state)
                & (df_trans.action == (row.hl_task[1], row.msg_budget_per_agent))
                & (df_trans.next_state == row.hl_task[1]),
                "prob",
            ] = 1.0 - row.test_project_failed_mean

            # fail rate
            df_trans.loc[
                (df_trans.state == row.hl_start_state)
                & (df_trans.action == (row.hl_task[1], row.msg_budget_per_agent))
                & (df_trans.next_state != row.hl_task[1]),
                "prob",
            ] = row.test_project_failed_mean

    def get_env_info(self):
        """standard function to interface with EPyMARL training loop"""
        env_info = {
            "state_shape": self._get_state_size(),
            # "obs_shape": self._get_obs_size(),
            # "n_actions": len(self.actions),
            # "n_agents": len(self.agents),
        }
        return env_info

    def _get_state_size(self) -> int:
        """standard function to interface with EPyMARL training loop,
        returns the flattened size of the global state."""
        # size of agent.shape
        state_size = int(np.prod(self.state.shape))
        return state_size

    def render(
        self,
        action: Optional[dict] = None,
        img_shape: tuple[float] = (6, 3),
    ):
        # make an image of the MDP using networkX to show the nodes + available edges between them
        # only set up the MDP graph once during training
        if self.graph is None:
            self.graph = nx.MultiDiGraph()

            for state in self.state_space:
                state_row = self._transition_probs.loc[
                    self._transition_probs.state == state
                ].iloc[0]

                self.graph.add_node(state, **{"state_type": state_row.state_type})

                # get all outgoing edges for this state
                df_edge = self._transition_probs.loc[
                    (self._transition_probs.state == state)
                ]

                for _, row in df_edge.iterrows():
                    self.graph.add_edges_from(
                        [
                            (
                                row.state,
                                row.next_state,
                                {"action": row.action},
                            ),
                        ]
                    )

            self.node_colors: list = []

            for node in self.graph.nodes:
                if self.graph.nodes[node]["state_type"] == "normal":
                    self.node_colors.append(self.colors["light_grey"])
                elif self.graph.nodes[node]["state_type"] == "fail":
                    self.node_colors.append(self.colors["red"])
                elif self.graph.nodes[node]["state_type"] == "goal":
                    self.node_colors.append(self.colors["yellow"])

        # add an outline to the agent's current state
        node_outline_colors: list = [self.colors["white"]] * len(self.graph.nodes)
        node_outline_widths: list = [self.edge_widths["normal"]] * len(self.graph.nodes)
        for i, node in enumerate(self.graph.nodes):
            if node == self.agent.state:
                node_outline_colors[i] = self.colors["black"]
                node_outline_widths[i] = self.edge_widths["highlight"]
                break

        # highlight chosen action
        edge_outline_colors: list = ["gray"] * len(self.graph.edges)
        edge_outline_widths: list = [self.edge_widths["normal"]] * len(self.graph.edges)
        if action is not None:
            action_tuple = self._get_action_tuple(action)
            for i, (*edge, attrs) in enumerate(self.graph.edges(keys=True, data=True)):
                if edge[0] == self.agent.state and attrs["action"] == action_tuple:
                    if edge[1] == self.fail_state:
                        edge_outline_colors[i] = "red"
                    else:
                        edge_outline_colors[i] = "green"
                    edge_outline_widths[i] = self.edge_widths["highlight"]

        fig, ax = plt.subplots(figsize=img_shape)

        # render the graph
        self._draw_labeled_multigraph(
            ax=ax,
            G=self.graph,
            edge_label="action",
            node_outline_colors=node_outline_colors,
            node_outline_widths=node_outline_widths,
            edge_outline_colors=edge_outline_colors,
            edge_outline_widths=edge_outline_widths,
        )

        img: NDArray = self._fig_to_array(fig)

        return img

    def _draw_labeled_multigraph(
        self,
        ax,
        G,
        edge_label: str,
        node_outline_colors: list,
        node_outline_widths: list,
        edge_outline_colors: list,
        edge_outline_widths: list,
    ):
        """
        https://networkx.org/documentation/stable/auto_examples/drawing/plot_multigraphs.html
        Length of connectionstyle must be at least that of a maximum number of edges
        between pair of nodes. This number is maximum one-sided
        for directed graph and maximum total connections for undirected graph.
        """
        # Works with arc3 and angle3 connectionstyles
        connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

        # spectral is a decent layout
        pos = nx.spectral_layout(G)
        # pos = nx.spring_layout(G, k=5/np.sqrt(G.order()))
        # pos = nx.planar_layout(G)
        # pos = nx.shell_layout(G)

        # draw nodes + labels
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=self.node_colors,
            edgecolors=node_outline_colors,
            linewidths=node_outline_widths,
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        # draw edges + labels
        labels = {}
        for *edge, attrs in G.edges(keys=True, data=True):
            labels[tuple(edge)] = f"a={attrs[edge_label]}"

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_outline_colors,
            width=edge_outline_widths,
            connectionstyle=connectionstyle,
            ax=ax,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            labels,
            connectionstyle=connectionstyle,
            label_pos=0.5,
            font_color="black",
            font_size=6,
            ax=ax,
        )

        # image formatting
        handles = [
            Line2D(
                [0],
                [0],
                color="white",
                marker="o",
                markerfacecolor=self.colors["yellow"],
                markersize=10,
                label="Project Success",
            ),
            Line2D(
                [0],
                [0],
                color="white",
                marker="o",
                markerfacecolor=self.colors["red"],
                markersize=10,
                label="Project Failure",
            ),
            # Circle((0, 0), radius=0.05, color=self.colors["yellow"], label="Project Success"),
            # Circle((0, 0), radius=0.05, color=self.colors["red"], label="Project Failure"),
            Line2D(
                [0],
                [0],
                color="green",
                lw=self.edge_widths["highlight"],
                label="Task Success",
            ),
            Line2D(
                [0],
                [0],
                color="red",
                lw=self.edge_widths["highlight"],
                label="Task Failure",
            ),
        ]

        plt.legend(handles=handles, fontsize=8)
        plt.box(False)
        plt.tight_layout()

    def _fig_to_array(self, fig: Figure) -> NDArray:
        """
        Convert matplotlib figure to numpy array (faster, in-memory method).

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure object

        Returns
        -------
        np.ndarray
            Image array with shape (height, width, 3) in RGB format
        """
        # Render figure to RGBA buffer
        fig.canvas.draw()

        # Get pixel buffer from canvas
        buf = fig.canvas.buffer_rgba()

        # Get figure dimensions
        w, h = fig.canvas.get_width_height()
        plt.close()

        # Reshape to (height, width, 4) for RGBA, then drop the alpha channel
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

        arr = arr[:, :, :3]

        return arr
