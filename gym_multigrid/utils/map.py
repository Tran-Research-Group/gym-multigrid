import numpy as np

from gym_multigrid.typing import Position


def distance_points(p1: Position, p2: Position, is_defeated: bool = False) -> float:
    """Calculate the squared distance of two points"""
    return (
        float(np.linalg.norm(np.array(p1) - np.array(p2)))
        if not is_defeated
        else float("inf")
    )


def distance_area_point(point: Position, area: list[Position]) -> float:
    """Calculate the squared distance of an area and a point"""
    distances = [np.linalg.norm(np.array(point) - np.array(node)) for node in area]
    return float(np.min(distances))
