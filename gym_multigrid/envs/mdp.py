import itertools as it
import copy
import pandas as pd
from typing import Literal, Optional
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from gymnasium import spaces, Env


class MDPAgent:

    # simple agent class for a simple MDP
    def __init__(self, init_state: int) -> None:
        self.state = init_state
        self.prev_state = init_state

    def reset(self, init_state: int):
        self.state = init_state
        self.prev_state = init_state


class ProjectMDP(Env):
    # currently only hard coded for LBF env
    """Using terminology from the project scheduling literature, this MDP represents a project which consists of multiple tasks with ordering (precedence) constraints, pre-defined transitions, and state-dependent action spaces."""

    def __init__(
        self,
        num_rooms: int,
        task_type: Literal["atomic", "composed"],
        num_comms_values: int,
    ):
        super().__init__()

        self.agent = MDPAgent(init_state=0)
        self.tasks: list[tuple]
        self.init_state: int
        self.goal_state: int
        self.fail_state: int
        self.state_space: NDArray[np.int_]
        self.successor_map: dict[tuple[int, tuple], int]
        self._build_env(
            num_rooms=num_rooms,
            task_type=task_type,
            num_comms_values=num_comms_values,
        )

        self.task_completed: bool = False

        # graph of the MDP for rendering
        self.graph: Optional[nx.Graph] = None
        self.node_colors: list[str]

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
        num_comms_values: int,
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

        self.state_space = np.arange(0, len(self.tasks) + 2)
        self.init_state = int(self.state_space[0])
        self.goal_state = int(self.state_space[-2])
        self.fail_state = int(self.state_space[-1])

        # include "stay" task for self-transition for all states
        for state in self.state_space:
            self.tasks.append((state.item(), state.item()))

        self.observation_space = spaces.Discrete(n=len(self.state_space))

        # Action space: 2D discrete-continuous vector (task_idx, comms_val in [0, 1])
        self.n_tasks = len(self.tasks)
        self.n_comms_values = num_comms_values
        self.action_space = spaces.Box(
            low=np.array([0, 0.0]),
            high=np.array([self.n_tasks - 1, 1.0]),
            dtype=np.float32,
        )

        # Discretize comms values from [0, 1] to n_comms_values levels
        self.comms_values = [i.item() for i in np.linspace(0, 1, num_comms_values)]

        # transition probs
        self.transition_probs: pd.DataFrame
        transition_probs: list[dict] = []

        # init probs as None until we have real data
        self.successor_map = {}
        for edge in self.tasks:
            curr_state, chosen_next_state = edge
            if curr_state == chosen_next_state:
                # add dummy actions for self-transition for goal state and fail state
                # dummy action for absorbing states always has a comms val of 0 since it isn't a real task
                action = (chosen_next_state, 0.0)
                match chosen_next_state:
                    case self.goal_state:
                        next_state_type = "goal"
                    case self.fail_state:
                        next_state_type = "fail"
                    case _:
                        next_state_type = "normal"

                self.successor_map[(curr_state, action)] = chosen_next_state
                transition_probs.append(
                    {
                        "state": curr_state,
                        "action": action,
                        "next_state": chosen_next_state,
                        "next_state_type": next_state_type,
                        "prob": 1.0,
                    }
                )

            else:
                for comms_val in self.comms_values:
                    action = (chosen_next_state, comms_val)

                    next_states = [chosen_next_state, self.fail_state]
                    next_state_types = [
                        "goal" if chosen_next_state == self.goal_state else "normal"
                    ]
                    next_state_types += ["fail"]

                    for next_state, next_state_type in zip(
                        next_states, next_state_types
                    ):
                        self.successor_map[(curr_state, action)] = chosen_next_state
                        transition_probs.append(
                            {
                                "state": curr_state,
                                "action": action,
                                "next_state": next_state,
                                "next_state_type": next_state_type,
                                "prob": None,
                            }
                        )

        self.transition_probs = pd.DataFrame.from_records(transition_probs)

    def step(
        self, action: dict, task_completed: bool, task_failed: bool
    ) -> tuple[int, float, bool, dict]:
        chosen_next_state = action["chosen_next_state"]
        comms_val_raw = action["comms_allocation"]

        # Due to how time steps work in the PYMARL runner, need to set task_completed
        # so it is seen in the get_state() method to populate pre_transition_data to be used to select actions
        self.task_completed = False

        if task_failed:
            next_state = self.fail_state

        else:
            if task_completed:
                self.task_completed = True

            # take action given by the hl agent
            # Discretize comms value to nearest level
            # only used if sampling from the action space for development purposes
            comms_val: float = self.comms_values[
                np.argmin(np.abs(np.array(self.comms_values) - comms_val_raw))
            ]
            action_tuple = (chosen_next_state, comms_val)
            next_state = self.successor_map[(self.agent.state, action_tuple)]

            # update agent state
            self.agent.prev_state = copy.deepcopy(self.agent.state)
            self.agent.state = next_state

        # Determine reward and termination
        terminated = self.agent.state == self.goal_state
        project_failed = self.agent.state == self.fail_state

        # reward = 1.0 if terminated else (-0.01 if failed else 0.0)
        reward = 0.0

        obs = self.get_state()
        env_info: dict = {"project_failed": project_failed}

        return (
            obs,
            reward,
            terminated,
            env_info,
        )

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
            Additional options (unused for now)

        Returns
        -------
        tuple[int, dict]
            (observation, info) where observation is the initial MDP state
        """
        super().reset(seed=seed)
        self.agent.reset(self.init_state)
        self.task_completed = False

        obs: NDArray = self.get_state()
        info: dict = {}

        return obs, info

    def get_state(self) -> NDArray:
        state = np.array([self.agent.state, self.task_completed])
        return state

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
        state = self.get_state()
        state_size = int(np.prod(state.shape))
        return state_size

    def render(self):
        # make an image of the MDP using networkX to show the nodes + available edges between them

        # TODO low priorirty - maybe be able to highlight the current chosen edge?
        # current actions don't work for that, would have to revisit that
        # current actions support "self transitions" so that wouldn't work nicely

        # only set up the graph
        if self.graph is None:
            self.graph = nx.MultiDiGraph()

            for state in self.state_space:
                df_state = self.transition_probs.loc[
                    (self.transition_probs.state == state)
                    & (self.transition_probs.next_state == state)
                ]

                state_type = df_state.next_state_type.item()
                self.graph.add_node(
                    int(state), **{"state_type": state_type, "current_state": False}
                )

                # get all outgoing edges for this state
                df_edge = self.transition_probs.loc[
                    (self.transition_probs.state == state)
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

            self.node_colors: list[str] = []

            for node in self.graph.nodes:
                if self.graph.nodes[node]["state_type"] == "normal":
                    self.node_colors.append("cyan")
                elif self.graph.nodes[node]["state_type"] == "fail":
                    self.node_colors.append("red")
                elif self.graph.nodes[node]["state_type"] == "goal":
                    self.node_colors.append("yellow")

        # add an outline to the agent's current state
        node_edge_colors: list[str] = []
        for i, node in enumerate(self.graph.nodes):
            if node == self.agent.state:
                node_edge_colors.append("black")
            else:
                node_edge_colors.append(self.node_colors[i])

        fig, ax = plt.subplots(figsize=(5, 3))

        # render the graph
        ax = self._draw_labeled_multigraph(
            G=self.graph, edge_label="action", node_edge_colors=node_edge_colors, ax=ax
        )

        img: NDArray = self._fig_to_array(fig)
        return img

    def _draw_labeled_multigraph(
        self, G, edge_label: str, node_edge_colors: list[str], ax=None
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
            G, pos, node_color=self.node_colors, edgecolors=node_edge_colors, ax=ax
        )
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        # draw edges + labels
        labels = {}
        for *edge, attrs in G.edges(keys=True, data=True):
            labels[tuple(edge)] = f"a={attrs[edge_label]}"

        nx.draw_networkx_edges(
            G, pos, edge_color="gray", connectionstyle=connectionstyle, ax=ax
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
        plt.box(False)
        plt.tight_layout()
        return ax
        # plt.savefig("hlmdp.png", dpi=200)

    def _fig_to_array(self, fig: plt.Figure) -> NDArray:
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

        # Reshape to (height, width, 4) for RGBA, then drop the alpha channel
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

        arr = arr[:, :, :3]

        return arr
