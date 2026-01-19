from heapq import heapify, heappop, heappush
from typing import NamedTuple, Union, Any, Type
import enum

import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import WorldObjT, WorldObj
from gym_multigrid.core.world import World


class AStarNode(NamedTuple):
    f: int
    g: int
    h: int
    parent: Union["AStarNode", None]
    loc: tuple[int, int]


def a_star(
    start: tuple[int, int],
    end: tuple[int, int],
    grid: Grid,
) -> list[tuple[int, int]]:
    """
    Compute the path from start to end using A* algorithm.

    Parameters
    ----------
    start : tuple[int,int]
        Start position
    end : tuple[int,int]
        End position
    grid : Grid
        Grid object containing the environment

    Returns
    -------
    path: list[tuple[int,int]]
        List of positions from start to end
    """

    # Add the start and end nodes
    start_node = AStarNode(
        manhattan_distance(start, end), 0, manhattan_distance(start, end), None, start
    )
    # Initialize and heapify the lists
    open_nodes: list[AStarNode] = [start_node]
    closed_nodes: list[AStarNode] = []
    heapify(open_nodes)
    path: list[tuple[int, int]] = []  # return of the func

    while open_nodes:
        # Get the current node popped from the open list
        current_node = heappop(open_nodes)

        # Push the current node to the closed list
        closed_nodes.append(current_node)

        # When the goal is found
        if current_node.loc == end:
            current: AStarNode | None = current_node
            while current is not None:
                path.append(current.loc)
                current = current.parent

            path.reverse()
            break

        else:
            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                # Get node location
                current_loc: tuple[int, int] = current_node.loc
                new_loc = (current_loc[0] + direction[0], current_loc[1] + direction[1])

                # Make sure within a range and avoid obstacles or other agents
                if (
                    loc_inside_map(new_loc, grid)
                    and loc_can_overlap(new_loc, grid)
                    or new_loc == end
                ):
                    # Create the f, g, and h values
                    g = current_node.g + 1
                    h = manhattan_distance(new_loc, end)
                    f = g + h

                    # Check if the new node is in the open or closed list
                    open_indices = [
                        i
                        for i, open_node in enumerate(open_nodes)
                        if open_node.loc == new_loc
                    ]
                    closed_indices = [
                        i
                        for i, closed_node in enumerate(closed_nodes)
                        if closed_node.loc == new_loc
                    ]

                    # Compare f values if the new node is already existing in either list
                    if closed_indices:
                        closed_index = closed_indices[0]
                        if f < closed_nodes[closed_index].f:
                            closed_nodes.pop(closed_index)
                            heappush(
                                open_nodes, AStarNode(f, g, h, current_node, new_loc)
                            )
                        else:
                            continue

                    elif open_indices:
                        open_index = open_indices[0]
                        if f < open_nodes[open_index].f:
                            open_nodes.pop(open_index)
                            open_nodes.append(AStarNode(f, g, h, current_node, new_loc))
                            heapify(open_nodes)
                        else:
                            continue

                    else:
                        heappush(open_nodes, AStarNode(f, g, h, current_node, new_loc))

                else:
                    continue

    if not path:
        raise ValueError("No path found")
    else:
        pass

    return path


def manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    """
    Compute a Manhattan distance of two points

    Parameters
    ----------
    p1: tuple[int,int]
        Location
    p2 : tuple[int,int]
        Another location

    Returns
    -------
    distance : int
        Manhattan distance between two points
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def loc_inside_map(loc: tuple[int, int], grid: Grid) -> bool:
    """
    Check if a location is inside the grid

    Parameters
    ----------
    loc: tuple[int,int]
        Location
    grid: Grid
        Grid object

    Returns
    -------
    inside: bool
        True if the location is inside the grid
    """
    x, y = loc
    return 0 <= x < grid.width and 0 <= y < grid.height


def loc_can_overlap(
    loc: tuple[int, int],
    grid: Grid,
) -> bool:
    """
    Check if a location can overlap with other objects.
    If the location is not empty and not in the list of avoided objects, it can overlap.
    In case the location is the end location, it can overlap regardless of the object at the location.


    Parameters
    ----------
    loc: tuple[int,int]
        Location
    grid: Grid
        Grid object

    Returns
    -------
    can_overlap: bool
        True if the location can overlap with other objects
    """
    grid_cell: WorldObjT | None = grid.get(*loc)
    can_overlap: bool

    match grid_cell:
        case WorldObj():
            can_overlap = grid_cell.can_overlap()
        case None:
            can_overlap = True
        case _:
            raise ValueError(f"Invalid object type: {type(grid_cell)}")

    return can_overlap


def extract_buildings_from_obs(
    obs: NDArray[np.int_], world: World
) -> list[dict[str, Any]]:
    """
    Extract all buildings from observation with their state information.

    Parameters
    ----------
    obs : NDArray[np.int_]
        Observation array of shape (width, height, 6)
    world : World
        SaveTheCityWorld instance for index lookups

    Returns
    -------
    buildings : list[dict[str, Any]]
        List of building dictionaries with keys:
        - 'pos': (x, y) position
        - 'type': 'fast_burning_building' or 'slow_burning_building'
        - 'building_state': int (0-100)
        - 'fire_rate': int (>=0)
        - 'alive': bool (building_state > 0)
    """
    buildings = []
    width, height = obs.shape[0], obs.shape[1]

    for x in range(width):
        for y in range(height):
            obj_type_idx = obs[x, y, 0]

            # Check if it's a building
            if obj_type_idx == world.OBJECT_TO_IDX.get("fast_burning_building"):
                buildings.append({
                    'pos': (x, y),
                    'type': 'fast_burning_building',
                    'building_state': int(obs[x, y, 2]),
                    'fire_rate': int(obs[x, y, 3]),
                    'alive': obs[x, y, 2] > 0
                })
            elif obj_type_idx == world.OBJECT_TO_IDX.get("slow_burning_building"):
                buildings.append({
                    'pos': (x, y),
                    'type': 'slow_burning_building',
                    'building_state': int(obs[x, y, 2]),
                    'fire_rate': int(obs[x, y, 3]),
                    'alive': obs[x, y, 2] > 0
                })

    return buildings


def extract_agents_from_obs(
    obs: NDArray[np.int_], world: World
) -> list[dict[str, Any]]:
    """
    Extract all agent positions from observation.

    Parameters
    ----------
    obs : NDArray[np.int_]
        Observation array of shape (width, height, 6)
    world : World
        SaveTheCityWorld instance for index lookups

    Returns
    -------
    agents : list[dict[str, Any]]
        List of agent dictionaries with keys:
        - 'pos': (x, y) position
        - 'type': 'firefighter', 'builder', or 'generalist'
    """
    agents = []
    width, height = obs.shape[0], obs.shape[1]

    for x in range(width):
        for y in range(height):
            obj_type_idx = obs[x, y, 0]

            if obj_type_idx == world.OBJECT_TO_IDX.get("firefighter"):
                agents.append({'pos': (x, y), 'type': 'firefighter'})
            elif obj_type_idx == world.OBJECT_TO_IDX.get("builder"):
                agents.append({'pos': (x, y), 'type': 'builder'})
            elif obj_type_idx == world.OBJECT_TO_IDX.get("generalist"):
                agents.append({'pos': (x, y), 'type': 'generalist'})

    return agents


def get_adjacent_positions(pos: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Get all 4 adjacent positions (N, E, S, W).

    Parameters
    ----------
    pos : tuple[int, int]
        Current position (x, y)

    Returns
    -------
    adjacent : list[tuple[int, int]]
        List of 4 adjacent positions
    """
    x, y = pos
    return [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]  # N, E, S, W


def is_building_occupied(
    building_pos: tuple[int, int], agent_positions: list[tuple[int, int]]
) -> bool:
    """
    Check if a building is already occupied by a teammate.
    A building is considered occupied if any agent is adjacent to it.

    Parameters
    ----------
    building_pos : tuple[int, int]
        Position of the building to check
    agent_positions : list[tuple[int, int]]
        List of all agent positions in the environment

    Returns
    -------
    occupied : bool
        True if any agent is adjacent to the building
    """
    adjacent_positions = get_adjacent_positions(building_pos)
    return any(agent_pos in adjacent_positions for agent_pos in agent_positions)


def find_nearest_fire(
    agent_pos: tuple[int, int],
    obs: NDArray[np.int_],
    world: World,
    exclude_occupied: bool = True
) -> tuple[int, int] | None:
    """
    Find the nearest building with fire, optionally excluding occupied ones.

    Parameters
    ----------
    agent_pos : tuple[int, int]
        Agent's current position
    obs : NDArray[np.int_]
        Observation array
    world : World
        SaveTheCityWorld instance
    exclude_occupied : bool
        If True, exclude buildings with adjacent teammates

    Returns
    -------
    nearest_fire_pos : tuple[int, int] | None
        Position of nearest fire, or None if no fires exist
    """
    # Extract all buildings
    buildings = extract_buildings_from_obs(obs, world)

    # Filter for burning buildings that are alive
    burning_buildings = [
        b for b in buildings
        if b['fire_rate'] > 0 and b['alive']
    ]

    if not burning_buildings:
        return None

    # If excluding occupied, filter out buildings with adjacent teammates
    if exclude_occupied:
        agents = extract_agents_from_obs(obs, world)
        # Exclude the current agent's position from occupancy check
        agent_positions = [a['pos'] for a in agents if tuple(a['pos']) != tuple(agent_pos)]

        burning_buildings = [
            b for b in burning_buildings
            if not is_building_occupied(b['pos'], agent_positions)
        ]

    if not burning_buildings:
        return None

    # Find nearest by Manhattan distance
    nearest = min(
        burning_buildings,
        key=lambda b: manhattan_distance(agent_pos, b['pos'])
    )

    return nearest['pos']


def find_nearest_incomplete_building(
    agent_pos: tuple[int, int],
    obs: NDArray[np.int_],
    world: World,
    exclude_occupied: bool = True
) -> tuple[int, int] | None:
    """
    Find the nearest building that is not completed (building_state < 100).

    Parameters
    ----------
    agent_pos : tuple[int, int]
        Agent's current position
    obs : NDArray[np.int_]
        Observation array
    world : World
        SaveTheCityWorld instance
    exclude_occupied : bool
        If True, exclude buildings with adjacent teammates

    Returns
    -------
    nearest_building_pos : tuple[int, int] | None
        Position of nearest incomplete building, or None if all complete
    """
    # Extract all buildings
    buildings = extract_buildings_from_obs(obs, world)

    # Filter for incomplete, non-burning buildings that are alive
    # IMPORTANT: Only target non-burning buildings (builders can't build while fire is active)
    incomplete_buildings = [
        b for b in buildings
        if b['building_state'] < 100 and b['fire_rate'] == 0 and b['alive']
    ]

    if not incomplete_buildings:
        return None

    # If excluding occupied, filter out buildings with adjacent teammates
    if exclude_occupied:
        agents = extract_agents_from_obs(obs, world)
        # Exclude the current agent's position from occupancy check
        agent_positions = [a['pos'] for a in agents if tuple(a['pos']) != tuple(agent_pos)]

        incomplete_buildings = [
            b for b in incomplete_buildings
            if not is_building_occupied(b['pos'], agent_positions)
        ]

    if not incomplete_buildings:
        return None

    # Find nearest by Manhattan distance
    nearest = min(
        incomplete_buildings,
        key=lambda b: manhattan_distance(agent_pos, b['pos'])
    )

    return nearest['pos']


def find_nearest_target_building(
    agent_pos: tuple[int, int],
    obs: NDArray[np.int_],
    world: World,
    prioritize_fire: bool = True,
    exclude_occupied: bool = True
) -> tuple[tuple[int, int] | None, str]:
    """
    Find the nearest building that needs attention (burning OR incomplete).
    Used by GeneralistPolicy.

    Parameters
    ----------
    agent_pos : tuple[int, int]
        Agent's current position
    obs : NDArray[np.int_]
        Observation array
    world : World
        SaveTheCityWorld instance
    prioritize_fire : bool
        If True, always choose fires over incomplete buildings
    exclude_occupied : bool
        If True, exclude buildings with adjacent teammates

    Returns
    -------
    target_pos : tuple[int, int] | None
        Position of target building
    target_type : str
        'fire', 'incomplete', or 'none'
    """
    # Extract all buildings
    buildings = extract_buildings_from_obs(obs, world)

    # Separate into fires and incomplete
    fires = [
        b for b in buildings
        if b['fire_rate'] > 0 and b['alive']
    ]
    incomplete = [
        b for b in buildings
        if b['building_state'] < 100 and b['fire_rate'] == 0 and b['alive']
    ]

    # If excluding occupied, filter both lists
    if exclude_occupied:
        agents = extract_agents_from_obs(obs, world)
        # Exclude the current agent's position from occupancy check
        agent_positions = [a['pos'] for a in agents if tuple(a['pos']) != tuple(agent_pos)]

        fires = [
            b for b in fires
            if not is_building_occupied(b['pos'], agent_positions)
        ]
        incomplete = [
            b for b in incomplete
            if not is_building_occupied(b['pos'], agent_positions)
        ]

    # If prioritize_fire and fires exist, return nearest fire
    if prioritize_fire and fires:
        nearest_fire = min(
            fires,
            key=lambda b: manhattan_distance(agent_pos, b['pos'])
        )
        return nearest_fire['pos'], 'fire'

    # Otherwise, combine both lists and find nearest overall
    all_targets = fires + incomplete

    if not all_targets:
        return None, 'none'

    nearest = min(
        all_targets,
        key=lambda b: manhattan_distance(agent_pos, b['pos'])
    )

    # Determine type
    target_type = 'fire' if nearest['fire_rate'] > 0 else 'incomplete'

    return nearest['pos'], target_type


def direction_to_action(
    from_pos: tuple[int, int],
    to_pos: tuple[int, int],
    action_set: Type[enum.IntEnum]
) -> int:
    """
    Convert movement from one position to another into an action.

    Parameters
    ----------
    from_pos : tuple[int, int]
        Current position
    to_pos : tuple[int, int]
        Next position
    action_set : Type[enum.IntEnum]
        SaveTheCityActions enum

    Returns
    -------
    action : int
        Action enum value (NORTH, EAST, SOUTH, WEST, or STILL)
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    if dx == 0 and dy == -1:  # Moving North (y decreases)
        return action_set.NORTH
    elif dx == 1 and dy == 0:  # Moving East (x increases)
        return action_set.EAST
    elif dx == 0 and dy == 1:  # Moving South (y increases)
        return action_set.SOUTH
    elif dx == -1 and dy == 0:  # Moving West (x decreases)
        return action_set.WEST
    else:  # No movement or invalid
        return action_set.STILL


def is_adjacent_to_target(
    agent_pos: tuple[int, int],
    target_pos: tuple[int, int]
) -> bool:
    """
    Check if agent is adjacent to target (Manhattan distance = 1).

    Parameters
    ----------
    agent_pos : tuple[int, int]
        Agent position
    target_pos : tuple[int, int]
        Target position

    Returns
    -------
    adjacent : bool
        True if Manhattan distance is 1
    """
    return manhattan_distance(agent_pos, target_pos) == 1


def is_adjacent_to_any_building(
    agent_pos: tuple[int, int],
    obs: NDArray[np.int_],
    world: World
) -> bool:
    """
    Check if agent is adjacent to any building.

    Parameters
    ----------
    agent_pos : tuple[int, int]
        Agent position
    obs : NDArray[np.int_]
        Observation array
    world : World
        SaveTheCityWorld instance

    Returns
    -------
    adjacent : bool
        True if agent is adjacent to any building
    """
    buildings = extract_buildings_from_obs(obs, world)
    for building in buildings:
        if is_adjacent_to_target(agent_pos, building['pos']):
            return True
    return False


def get_valid_move_away_action(
    agent_pos: tuple[int, int],
    obs: NDArray[np.int_],
    world: World,
    action_set: Type[enum.IntEnum],
    random_generator: np.random.Generator
) -> int:
    """
    Get a random valid movement action that doesn't go through walls/buildings.

    Note: Does not check for agent collisions since agents can overlap and
    coordinating movement between agents computing actions simultaneously
    would add significant complexity.

    Parameters
    ----------
    agent_pos : tuple[int, int]
        Agent position
    obs : NDArray[np.int_]
        Observation array
    world : World
        SaveTheCityWorld instance
    action_set : Type[enum.IntEnum]
        SaveTheCityActions enum
    random_generator : np.random.Generator
        Random number generator

    Returns
    -------
    action : int
        A valid movement action or STILL if no valid moves
    """
    width, height = obs.shape[0], obs.shape[1]

    # Possible movement directions: N, E, S, W
    directions = [
        (action_set.NORTH, (0, -1)),
        (action_set.EAST, (1, 0)),
        (action_set.SOUTH, (0, 1)),
        (action_set.WEST, (-1, 0)),
    ]

    # Indices of static objects that block movement
    blocked_indices = [
        world.OBJECT_TO_IDX.get("wall"),
        world.OBJECT_TO_IDX.get("fast_burning_building"),
        world.OBJECT_TO_IDX.get("slow_burning_building"),
    ]

    valid_actions = []
    for action, (dx, dy) in directions:
        new_pos = (agent_pos[0] + dx, agent_pos[1] + dy)

        # Check bounds
        if not (0 <= new_pos[0] < width and 0 <= new_pos[1] < height):
            continue

        # Check if position is passable (not wall, not building)
        obj_type_idx = obs[new_pos[0], new_pos[1], 0]
        if obj_type_idx not in blocked_indices:
            valid_actions.append(action)

    if valid_actions:
        return random_generator.choice(valid_actions)
    return action_set.STILL


def obs_to_grid(obs: NDArray[np.int_], world: World) -> Grid:
    """
    Convert observation array to Grid object for A* pathfinding.

    This reconstructs a Grid from observation data, which is the same
    information an RL agent would have access to.

    Parameters
    ----------
    obs : NDArray[np.int_]
        Observation array of shape (width, height, 6)
    world : World
        SaveTheCityWorld instance for object creation

    Returns
    -------
    grid : Grid
        Grid object populated with walls and buildings
    """
    from gym_multigrid.core.object import Wall, Building

    width, height = obs.shape[0], obs.shape[1]
    grid = Grid(width, height, world)

    for x in range(width):
        for y in range(height):
            obj_type_idx = obs[x, y, 0]
            if obj_type_idx == world.OBJECT_TO_IDX.get("wall"):
                grid.set(x, y, Wall(world))
            elif obj_type_idx == world.OBJECT_TO_IDX.get("fast_burning_building"):
                grid.set(x, y, Building(world, burn_rate=0, build_speed=0, firefight_speed=0, fast_burning=True))
            elif obj_type_idx == world.OBJECT_TO_IDX.get("slow_burning_building"):
                grid.set(x, y, Building(world, burn_rate=0, build_speed=0, firefight_speed=0, fast_burning=False))

    return grid

    return grid
