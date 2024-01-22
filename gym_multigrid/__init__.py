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
# Collect game with 1 agent and 3 object types
# ----------------------------------------
register(
    id="multigrid-collect-single-v0",
    entry_point="gym_multigrid.envs:CollectGame3ObjSingleAgent",
)

register(
    id="multigrid-collect-rooms-v0",
    entry_point="gym_multigrid.envs:CollectGameRooms",
)

register(
    id="multigrid-collect-rooms-fixed-horizon-v0",
    entry_point="gym_multigrid.envs:CollectGameRoomsFixedHorizon",
)

register(
    id="multigrid-collect-rooms-respawn-v0",
    entry_point="gym_multigrid.envs:CollectGameRoomsRespawn",
)
