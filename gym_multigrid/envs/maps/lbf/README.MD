# Field map usage

- Use VS Code's "Edit CSV" extension to edit these maps easily.

- You cannot choose the team's inital state in the first room or the fruit spawn locations, the env handles those automatically.

- goal
    -  (g, room_idx) = any agent can reach this goal, the room's task ends when all agents reach a goal
    - (g, room_idx, agent_idx) = a goal state assigned to agent with index agent_idx

- (d, room_idx) = door (wall) that disappears when agents reach the goal states after completing all other tasks in room with index room_idx

- w = permanent wall
