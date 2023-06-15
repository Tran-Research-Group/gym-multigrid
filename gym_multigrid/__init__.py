from gymnasium.envs.registration import register


# Collect game with 2 agents and 3 object types
# ----------------------------------------
register(
    id="multigrid-collect-v0",
    entry_point="multigrid.envs:CollectGame3Obj2Agent",
)
# Collect game with 2 agents and 3 object types with fixed locations
# ----------------------------------------
register(
    id="multigrid-collect-fixed-v0",
    entry_point="multigrid.envs:CollectGame3ObjFixed2Agent",
)
