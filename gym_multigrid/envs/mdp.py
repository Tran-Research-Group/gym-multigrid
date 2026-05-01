import copy
import pandas as pd
from typing import Literal, Optional
import networkx as nx
import matplotlib.pyplot as plt

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

    def render(self):
        # low priority, get other things working first
        pass

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
        # make an image of the MDP
        ## use networkX to show the nodes + available edges between them
        # hl_img = np.zeros_like(ll_img)

        # init networkx graph for visualization
        # you only have to do this once, otherwise you can update the coloring of a given node w/ the current state or whatever
        ## all normal nodes have black outlines
        ## fail node has red
        ## if an agent in the node, it has a little circle in it

        ## show all edges (next state + comms allocations)
        ## low priorirty - maybe be able to highlight the current chosen edge?
        ### current actions don't work for that, would have to revisit that
        ### current actions support "self transitions" so that wouldn't work nicely

        graph = nx.MultiDiGraph()
        edge_labels = {}

        success_edges = []
        fail_edges = []

        for state in self.state_space:
            df_tmp = self.transition_probs.loc[self.transition_probs.state == state]
            for _, row in df_tmp.iterrows():
                graph.add_edges_from(
                    [
                        (
                            row.state,
                            row.next_state,
                            {"action": row.action, "edge_type": row.next_state_type},
                        ),
                    ]
                )

        fig, ax = plt.subplots(figsize=(6, 4))
        # plot the graph
        pos = nx.planar_layout(graph)
        # pos = nx.spring_layout(graph, k=0.15, iterations=20)
        # pos = nx.spectral_layout(graph)

        nx.draw(graph, pos)
        import os

        plt.box(False)
        plt.tight_layout()
        save_path = os.path.join(f"hlmdp.png")
        plt.savefig(save_path, dpi=200)
        print("\n breakpoint ")
        __import__("ipdb").set_trace(context=3)

        # # populate edge data from the high-level MDP solution
        # graph.add_node(self.env.u_fail)

        # for u in self.env.state_space:
        #     if u != self.env.u_fail:
        #         graph.add_node(u)

        #         if u != self.env.u_goal:
        #             for action in self.env.avail_actions[u]:
        #                 u_next = self.env.successor[u, action]

        #                 # only show the action index on the edges
        #                 # edge_labels[(u, final_state)] = f"a: {action}"

        #                 # # show a bunch on information on the edges (gets cut off b/c some edges are too short)
        #                 # state_action_occ_str = (
        #                 #     "$x$"
        #                 #     + f"$(u={u}, u'={action})$: {round(self.opt_vars.state_action_occupancy[u, action].x, 3)}\n"
        #                 # )
        #                 # policy_str = (
        #                 #     "$\pi$"
        #                 #     + f"$(u'={action}|u={u})$: {round(self.policy[u, action], 3)}\n"
        #                 # )
        #                 # optimal_comms_val_str = f"$\lambda_{action}^*$: {round(self.optimal_comms_vals[action], 3)}\n"
        #                 # success_prob_str = (
        #                 #     "$\hat{p}_{u u' w}$"
        #                 #     + f"($\lambda_{action}^*$): {round(self.chosen_success_probs[action], 3)}"
        #                 # )
        #                 # edge_labels[(u, u_next)] = (
        #                 #     state_action_occ_str
        #                 #     + policy_str
        #                 #     + optimal_comms_val_str
        #                 #     + success_prob_str
        #                 # )
        #                 # graph.add_edge(u, u_next, action_idx=action)

        #                 # add edges from all start states to a fail state
        #                 # success_prob_str = "$\hat{p}_{u u' w}$" + f"($\lambda_{action}^*$): {round(1.0 - self.chosen_success_probs[action], 3)}"
        #                 # edge_labels[(u, self.env.u_fail)] = success_prob_str
        #                 graph.add_edge(u, self.env.u_fail, action_idx=action)

        # fig, ax = plt.subplots(figsize=(6, 4))

        # # place a text box in upper left in axes coords
        # text_str = (
        #     f"Specified Goal Reach Probability: {self.success_prob_spec}\n"
        #     + f"Chosen Goal Reach Probability: {round(self.goal_reach_prob, 3)}\n"
        #     + f"Total Comms. Cost: {round(self.get_objective_value(), 3)}"
        # )
        # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        # ax.text(
        #     0.02,
        #     0.98,
        #     text_str,
        #     transform=ax.transAxes,
        #     fontsize=8,
        #     verticalalignment="top",
        #     bbox=props,
        # )

        # # plot the graph
        # pos = nx.planar_layout(graph)
        # # pos = nx.spring_layout(graph, k=0.15, iterations=20)
        # # pos = nx.spectral_layout(graph)
        # node_labels = {}
        # for node in graph.nodes:
        #     if node != self.env.u_fail:
        #         node_labels[node] = node
        #     else:
        #         node_labels[node] = "x"

        # nx.draw_networkx_nodes(graph, pos)
        # nx.draw_networkx_labels(graph, pos, labels=node_labels)

        # edges = list(graph.edges())
        # success_edges = []
        # fail_edges = []
        # for e in edges:
        #     if e[1] != self.env.u_fail:
        #         success_edges.append(e)
        #     else:
        #         fail_edges.append(e)

        # nx.draw_networkx_edges(graph, pos, edgelist=success_edges, edge_color="k")
        # nx.draw_networkx_edges(graph, pos, edgelist=fail_edges, edge_color="r")
        # nx.draw_networkx_edge_labels(
        #     graph,
        #     pos,
        #     label_pos=0.5,
        #     edge_labels=edge_labels,
        #     font_size=7,
        #     rotate=False,
        # )
        # plt.box(False)
        # plt.tight_layout()
        # save_dir = os.path.join(
        #     self.eval_dir,
        #     "hlm_visualization",
        #     self.comms_formulation,
        # )
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(
        #     save_dir, f"hlm__succes_spec_{self.success_prob_spec}.png"
        # )
        # plt.savefig(save_path, dpi=200)
        # plt.close(fig)

        # # print("\n\n")
        # # print(f"Policy: \n{self.policy}\n")
        # # print(f"Chosen success probabilities: {self.chosen_success_probs}")
        # # print(f"Communication values: {self.optimal_comms_vals}")
        # # print(f"Summed communication values: {sum(self.optimal_comms_vals.values())}")
        # # print(f"Goal reach probability: {self.goal_reach_prob}")
