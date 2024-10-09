# Capture the Flag (CtF)

## Overview
The CtF game is a simple grid-world environment with discrete state and action spaces, but includes complex adversarial dynamics.

## Game Dynamics

In our CtF game, if a pair of blue and red agents are next to each other in the blue territory, then the red agent is killed with 75\% probability (and vice versa in the red territory).
The game ends when either agent captures its opponent's flag or all blue agents are defeated. 
The 12 $\times$ 12 state space is fully observable, and there are 5 discrete actions for an agent: stay, up, right, down, and left.
The field objects consist of the $m$ friendly agents, $n$ enemy agents, 2 flags, and 96 territories (48 for each agent), and 48 obstacles (4 at the center and 44 surrounding the the territories).
The observation of a state is a 12 $\times$ 12 $\times$ 3 tensor, where the first layer represents a map of the territories and obstacles, the second layer represent a map of the agents and flags, and the third layer represents status of the agents whether an agent is dead or alive.
This observation tensor is used as the input to the RL algorithms.

At every timestep, the blue agent is rewarded -0.01 to encourage it to reach the red flag faster.
The blue agent is rewarded 1 by capturing the red flag and -1 by having the blue flag captured by the red agent.
Furthermore, the blue agent is rewarded 0.25 by killing the red agent and -0.25 by being killed by the red agent.

## M vs. N CtF Game 
The default CtF game.
| Attribute             | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| Action Space          | `MultiDiscrete(5)`                                                                   |
| Observation Space     | `positional`, `map`, `flattened`, `pos_map`, `pos_map_flattened`, `tensor`           |
| Observation Encoding  | See [Observation Options](#observation-options)                                      |
| Reward                | `flag_reward`, `battle_reward_ratio`, `obstacle_penalty_ratio`, `step_penalty_ratio` |
| Number of Agents      | `m` blue (friendly, controlled) agents, `n` red (enemy, uncontrolled) agents         |
| Termination Condition | When all the blue agents are killed, or either of the flags is captured              |
| Truncation Steps      | `100`                                                                                |
| Creation              | Directly import the class                                                            |

## 1 vs. 1 CtF Game
Wrapper of M vs. N CtF game when $M = 1$ and $N = 1$.
| Attribute             | Description                                                                          |
| --------------------- | ------------------------------------------------------------------------------------ |
| Action Space          | `Discrete(5)`                                                                        |
| Observation Space     | `positional`, `map`, `flattened`, `pos_map`, `pos_map_flattened`, `tensor`           |
| Observation Encoding  | See [Observation Options](#observation-options)                                      |
| Reward                | `flag_reward`, `battle_reward_ratio`, `obstacle_penalty_ratio`, `step_penalty_ratio` |
| Termination Condition | When the blue agent is killed, or either of the flags is captured                    |
| Number of Agents      | 1 blue (friendly, controlled) agent, 1 red (enemy, uncontrolled) agent               |
| Truncation Steps      | `100`                                                                                |
| Creation              | Directly import the class                                                            |

## Observation Options
There are five options for the observation:  `positional`, `map`, `flattened`, `pos_map`, `pos_map_flattened`, `tensor`.

### `positional`
The `positional` observation provides dictionary of object positions in the map ($2 \times \{\text{number of objects}\}$) in addition to binary flags of agents being alive or dead, whose size is $m+n$ in M vs N CtF game while $1$ for the 1v1 game.

### `map`
The `map` observation is an encoded map which is `np.NDArray` of the size of the map.

### `flattened`
The `flattened` observation is a flattened vector version of `positional`.

### `pos_map`
The `pos_map` observation is a dictionary observation of the agent and flag positions, encoded map of the static objects (obstacles, blue & red territories) with the dead/alive flags.

### `pos_map_flattened`
The `pos_map_flattened` observation is a flattened vector version of `pos_map`.

### `tensor`
The `tensor` observation is a feature tensor whose size is map height $\times$ map width $\times$ 3, where the first layer represents a map of the territories and obstacles, the second layer represent a map of the agents and flags, and the third layer represents status of the agents whether an agent is dead or alive.
