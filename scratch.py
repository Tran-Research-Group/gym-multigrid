import pdb
from gym import register


from scratch_2 import EnvObjectGroup, RewardFunctions, TerminationFunctions, StateData, SubtaskData, HLMDPData, EnvData

"""
you need a data structure to manage the following situation:
for each subtask, you need to define a separate:
- the subtask index (this only serves to pick out the "correct" element from some collection associated with the given subtask, the env should not have any idea of this)
- initial state distribution
- set of terminal states (you only really need the reward function and termination function inside the labyrinth env)
- reward function
- "termination" function (check if the env terminated by reaching a goal state)
- _gen_grid function (becuase you can have different sets of objects spawned or not depending on the current subtask)


# within the labyrinth class, in step(), all I want to be able to do is call self._terminated

"""

####################
# HLMDP config
####################
# defining the subtasks should happen outside the env, not in a YAML config file, but hardcoded somewhere in python code
# the subtasks in HLMDPData should have everything they need to be passed into the env and fully specify a given Dec-POMDP for that subtask
## there can be no references to subtasks in the labyrinth or multigrid env class
## that means you need to define these functions outside the env classes, then read them in to overwrite the env's base reward, termination, and gen_grid functions when you instantiate it

state_data_list = [
    StateData(
        idx=0,
        outgoing_init_state_dist=[1.0, [[1, 2], [2, 1]]],
    )
]


subtask_data_list = [
    SubtaskData(
        edge=(0, 1),
        idx=0,
        final_state=[[2, 2], [2, 6]],
        termination_condition="reach_assigned_final_state",
    ),
    SubtaskData(
        edge=(0, 2),
        idx=1,
        final_state=[[3, 2], [3, 6]],
        termination_condition="reach_assigned_final_state",
    ),
]


# this should just have the basic stuff, like the structure of the env
## the idea is that this could be used for purposes other than specifying a Gym env
## this is basically like your automaton class in its scope
hlmdp_data = HLMDPData(state_data_list, subtask_data_list)

####################
# Dec-POMDP config
####################
# define your object groups for each subtask, then pass that in as a kwarg to Labyrinth
## and then you need to define Labyrinth's _gen_grid to loop over obj_group and spawn all the required objects


# this data structure is a lot like Miki's "object group", but I made it a dataclass so I can see its properties with intellisense
env_object_list = [
    EnvObjectGroup(obj_type="goal",
                   group_idx=0,
                   pos=((4, 1), (4, 2), (4, 3)),
                   color="green",
                   spawned_subtask_idxs=(0, 1, 2)
                )
]


# have a separate class that basically serves to act as an interface to an env, where the reward function, termination function, and env objects are brought in to the subtask data
env_data = EnvData(state_data=hlmdp_data.state_data,
                          subtask_data=hlmdp_data.subtask_data,
                          env_object_list=env_object_list,
                        #   reward_functions=RewardFunctions(),
                        #   termination_functions=TerminationFunctions()

)


pdb.set_trace()


# # data to specify each Dec-POMDP that models a subtask
# dec_pomdp_data = {}
# for _, data in hlmdp_data.values():
#     init_state_dist = data["outgoing_init_state_dist"]

#     for subtask_idx, subtask_data in data["subtask_idx"]:
#         final_state = subtask_data["final_state"]
#         termination_condition = subtask_data["termination_condition"]

#         dec_pomdp_data[subtask_idx] = {
#             "init_state_dist": init_state_dist,
#             "reward_function": RewardFunctions(termination_condition, final_state).reward_function,
#             "termination_function": TerminationFunctions(termination_condition, final_state).termination_function,
#             "env_objects": env_objects[subtask_idx]
#         }

pdb.set_trace()

####################

####################


# subtask_idx = 0

# # this basically runs the env's "init" function
# register(id="labyrinth_env", kwargs={
#     "init_state_dist": hlmdp_data.subtask_data[subtask_idx].init_state_dist,
#     }
# )


# you have to call "make" because there are some wrapper functions within gymnasium that make the envs more usable
# "make" serves to abstract those away
# # gymnasium.make("labyrinth_env",
#                )