from gymnasium.envs.registration import register


# Collect game with 2 agents and 3 object types
# ----------------------------------------
register(
    id="multigrid-collect-v0",
    entry_point="gym_multigrid.envs:CollectGame3Obj2Agent",
)
# Collect game with 2 agents and 3 object types with fixed locations
# ----------------------------------------
register(
    id="multigrid-collect-fixed-v0",
    entry_point="gym_multigrid.envs:CollectGame3ObjFixed2Agent",
)
# Collect game with single agent and 3 object types
# ----------------------------------------
register(
    id="multigrid-collect-single-v0",
    entry_point="gym_multigrid.envs:CollectGame3ObjSingleAgent",
)
# Collect game with 2 agents and 3 object types clustered in different quadrants of the grid
# ----------------------------------------
register(
    id="multigrid-collect-quadrants-v0",
    entry_point="gym_multigrid.envs:CollectGameQuadrants",
)
# Collect game with 2 agents and 3 object types clustered differently in four rooms
# ----------------------------------------
register(
    id="multigrid-collect-rooms-v0",
    entry_point="gym_multigrid.envs:CollectGameRooms",
)
