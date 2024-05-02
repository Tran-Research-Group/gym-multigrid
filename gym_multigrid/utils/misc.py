from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import random
from gym_multigrid.core.constants import STATE_IDX_TO_COLOR_WILDFIRE, TILE_PIXELS
from .rendering import fill_coords, point_in_circle, point_in_rect
from gym_multigrid.core.agent import Agent
from gym_multigrid.core.grid import Grid
from ..core.world import WorldT
from numpy.typing import NDArray


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def save_frames_as_gif(frames, path="./", filename="collect-", ep=0, fps=60, dpi=72):
    filename = filename + str(ep) + ".gif"
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=fps)
    plt.close()


def render_agent_tile(
    img: NDArray, agent: Agent, helper_grid: Grid, world: WorldT
) -> NDArray:
    """
    Render tile containing agent with background color corresponding to the state of tree in that cell.

    :param img: image of wildfire grid with trees missing at agent locations
    :param agent: agent
    :param helper_grid: helper grid containing all trees including missing trees
    :param world: wildfire world
    :return: image with all trees and agents rendered
    """

    pos = agent.pos
    o = helper_grid.get(*pos)
    s = o.state

    tile_size = TILE_PIXELS
    ymin = pos[1] * tile_size
    ymax = (pos[1] + 1) * tile_size
    xmin = pos[0] * tile_size
    xmax = (pos[0] + 1) * tile_size

    tree_color = world.COLORS[STATE_IDX_TO_COLOR_WILDFIRE[s]]
    fill_coords(
        img[ymin:ymax, xmin:xmax, :],
        point_in_circle(0.5, 0.5, 0.25),
        world.COLORS[agent.color],
        bg_color=tree_color,
    )
    fill_coords(
        img[ymin:ymax, xmin:xmax, :], point_in_rect(0, 0.031, 0, 1), (100, 100, 100)
    )
    fill_coords(
        img[ymin:ymax, xmin:xmax, :], point_in_rect(0, 1, 0, 0.031), (100, 100, 100)
    )
    return img


def render_rescue_tile(
    img: NDArray, pos: tuple, helper_grid: Grid, world: WorldT
) -> NDArray:
    """
    Render tile containing agent with background color corresponding to the state of tree in that cell.

    :param img: image of wildfire grid with trees missing at agent locations
    :param agent: agent
    :param helper_grid: helper grid containing all trees including missing trees
    :param world: wildfire world
    :return: image with all trees and agents rendered
    """

    o = helper_grid.get(*pos)
    s = o.state

    tile_size = TILE_PIXELS
    ymin = pos[1] * tile_size
    ymax = (pos[1] + 1) * tile_size
    xmin = pos[0] * tile_size
    xmax = (pos[0] + 1) * tile_size

    tree_color = world.COLORS[STATE_IDX_TO_COLOR_WILDFIRE[s]]
    fill_coords(
        img[ymin:ymax, xmin:xmax, :],
        point_in_circle(0.5, 0.5, 0.1),
        world.COLORS["black"],
        bg_color=tree_color,
    )
    fill_coords(
        img[ymin:ymax, xmin:xmax, :], point_in_rect(0, 0.031, 0, 1), (100, 100, 100)
    )
    fill_coords(
        img[ymin:ymax, xmin:xmax, :], point_in_rect(0, 1, 0, 0.031), (100, 100, 100)
    )
    return img


def get_central_square_coordinates(N, C):
    """
    Takes a grid size N and a square size C, and returns the coordinates of cells in a C x C square at the center of the grid.

    Args:
      N: The size of the grid (can be even or odd).
      C: The size of the square to be extracted.

    Returns:
      A list of tuples, where each tuple represents the (x, y) coordinates of a cell in the center square.

    Raises:
      ValueError: If C is larger than N.
    """

    if C > N:
        raise ValueError("Square size C cannot be larger than grid size N.")

    if N % 2 == 0:
        center_x = N // 2
        center_y = N // 2
        offset_x = (C - 1) // 2  # Offset adjustments for both even and odd squares
        offset_y = (C - 1) // 2

        start_x = center_x - offset_x
        start_y = center_y - offset_y

        coordinates = []
        for x in range(start_x, start_x + C):
            for y in range(start_y, start_y + C):
                coordinates.append((x, y))
    else:
        index = int((C - 1) / 2)
        start_x = int((N + 1) / 2)
        start_y = int((N + 1) / 2)

        coordinates = []
        for x in range(start_x - index, start_x + index + 1):
            for y in range(start_y - index, start_y + index + 1):
                coordinates.append((x, y))

    return coordinates


def get_nxn_square_coordinates(x, y, grid_size, n):
    """
    Takes the center cell position (x, y) and grid size, and returns the coordinates of all cells in a n x n square around the center.

    Args:
      x: The x-coordinate of the center cell if n is odd, or the x-coordinate of the top-left corner cell if n is even.
      y: The y-coordinate of the center cell if n is odd, or the y-coordinate of the top-left corner cell if n is even.
      grid_size: The size of the grid (must be greater than or equal to n).
      n: The size of the square to be extracted (must be odd).

    Returns:
      A list of tuples, where each tuple represents the (x, y) coordinates of a cell in the n x n square.

    Raises:
      ValueError: If grid size is less than n.
    """

    if grid_size < 3:
        raise ValueError("Grid size must be greater than or equal to 3.")

    if n % 2 == 0:
        coordinates = []
        for i in range(x, x + n):
            for j in range(y, y + n):
                coordinates.append((i, j))

        return coordinates
    else:
        # Calculate offsets for the n x n square
        offset = int(
            (n - 1) / 2
        )  # Since the square is centered, offset is (n-1)/2 in each direction

        # Ensure coordinates stay within grid boundaries
        start_x = max(1, x - offset)
        end_x = min(grid_size, x + offset)
        start_y = max(1, y - offset)
        end_y = min(grid_size, y + offset)

        coordinates = []
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                coordinates.append((x, y))

        return coordinates
