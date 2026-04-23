# Field map usage

- Use VS Code's "Edit CSV" extension to edit these maps easily.

- You can choose the team's inital state in the first room or the fruit spawn locations. If not specified in the map, the env will randomly spawn them.

- agent
    - (a, agent_level, agent_idx)

- fruit
    - (f, fruit_level)

- goal
    -  g = any agent can reach this goal, the room's task ends when all agents reach a goal
    - (g, agent_idx) = a goal state assigned to agent with index agent_idx

- door
    - d, wall that disappears when agents reach the goal states after completing all other tasks in room with index room_idx

- permanent wall
    - w
