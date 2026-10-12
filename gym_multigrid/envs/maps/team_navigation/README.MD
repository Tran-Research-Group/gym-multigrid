# Field map usage

- Use VS Code's "Edit CSV" extension to edit these maps easily.

- You can choose the team's inital state in the first room or the fruit spawn locations. If not specified in the map, the env will randomly spawn them.

- agent
    - (a, agent_level, agent_idx)

- goal
    -  g = any agent can reach this goal, the room's task ends when all agents reach a goal
    - (g, agent_idx) = a goal state assigned to agent with index agent_idx

- door
    - d, wall that disappears when agents reach the goal states after completing all other tasks in room with index room_idx

- permanent wall
    - w

## High-level navigation tasks

Hierarchical team navigation can load a YAML file through the `navigation_config`
environment argument. The file may define an arbitrary directed task graph under
`mdp.transitions` and one navigation task per directed edge under `tasks`.
Each task uses `from_state`, `to_state`, one goal position per agent, and an
`init_state_dist` when it is an initial task. Successor tasks may omit
`init_state_dist` when dependent-subtask training supplies a learned distribution
through reset options. See `gym_multigrid/envs/maps/team_navigation/navigation_branching.yaml`
for a branching example where two tasks merge into a final state, and
`navigation_hallway_learned.yaml` for a chain with root-only configuration.

By default, each goal is assigned to the agent with the same list index. A task
may instead set `goal_assignment: one_to_one` to make its `goal_positions` an
interchangeable set: any agent may occupy any goal, and the task succeeds only
when every goal is occupied by exactly one agent at the same time. The goal set
must still contain exactly one unique position per agent. Successful terminal
states then retain the actual agent-to-position assignment for learned
successor-task spawn distributions.

## High-level communication optimization

For independent low-level training, evaluate every configured `hl_task` at each
communication budget. The resulting `eval_stats` rows must include
`hl_start_state`, `hl_task`, `msg_budget_per_agent`,
`test_task_completed_mean`, and `test_n_episodes`. High-level optimization uses
the latest evaluation per run and pools its episode counts to estimate each
task/budget success probability. It then solves the DAG edge-occupancy problem
to minimize expected communication capacity subject to the requested
project-completion probability. The selected capacity is deterministic per
subtask; subtask selection can be stochastic.
Run this post-processing step with `post_processing=optimize_high_level_policy`,
the `maic_comm_opt` algorithm config, and the same `time_id` as the independent
low-level runs. The optimized task and capacity policies are written to
`results/high_level_policies/<time_id>/`.

When a navigation YAML explicitly defines transitions, its `goal_states` are
terminal states in the high-level MDP. Legacy environments without an explicit
transition graph retain the additional post-goal state behavior.
