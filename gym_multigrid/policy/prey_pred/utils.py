from heapq import heapify, heappop, heappush
from typing import Literal, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import WorldObj
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
        Start tuple[int,int]
    end : tuple[int,int]
        End tuple[int,int]
    field_map : NDArray[np.int_]
        Map of the environment
    world : World
        World object

    Returns
    -------
    path: list[tuple[int,int]]
        List of tuple[int,int]s from start to end
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
    Check if a location is inside the field_map

    Parameters
    ----------
    loc: tuple[int,int]
        Location
    field_map: NDArray[np.int_]
        Map of the environment

    Returns
    -------
    inside: bool
        True if the location is inside the field_map
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
    end_loc: tuple[int,int]
        End location
    field_map: NDArray[np.int_]
        Map of the environment
    world: World
        World object
    avoided_objects: list[str]
        List of objects to avoid

    Returns
    -------
    can_overlap: bool
        True if the location can overlap with other objects
    """
    grid_cell: WorldObj | None = grid.get(*loc)
    can_overlap: bool

    match grid_cell:
        case WorldObj():
            can_overlap = grid_cell.can_overlap()
        case None:
            can_overlap = True
        case _:
            raise ValueError(f"Invalid object type: {type(grid_cell)}")

    return can_overlap
