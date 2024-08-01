from matplotlib import animation
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from gym_multigrid.core.constants import STATE_IDX_TO_COLOR_WILDFIRE, TILE_PIXELS
from gym_multigrid.utils.rendering import fill_coords, point_in_circle, point_in_rect
from gym_multigrid.core.agent import Agent
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.world import WorldT


def save_frames_as_gif(
    frames, path="./", filename="env-render-episode-", ep=0, fps=60, dpi=72
):
    """Save a list of frames as a gif.

    Parameters
    ----------
    frames : ndarray
        list of frames to be saved as gif. Each frame is an RGB image of the environment at a time step.
    path : str, optional
       location to save the gif, by default "./"
    filename : str, optional
        name of the gif, by default "env-render-episode-". The episode number is appended to the filename. Thus, entire path of saved gif is "path/filename"+"ep.gif"
    ep : int, optional
        the episode number during which the frames were recorded, by default 0. For example, it is useful if the frames are saved during RL agent training to identify the episode number during which the frames were recorded.
    fps : int, optional
        frequency at which consecutive images or frames are displayed in the gif, by default 60
    dpi : int, optional
        output resolution of gif, by default 72
    """
    filename = filename + str(ep) + ".gif"
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=fps)
    plt.close()


def render_agent_tiles(
    img: NDArray,
    agent: Agent,
    helper_grid: Grid,
    world: WorldT,
    x_min: list[int] = None,
    y_min: list[int] = None,
    x_max: list[int] = None,
    y_max: list[int] = None,
    colors=None,
) -> NDArray:
    """
    Re-render the tile containing given agent to add background color corresponding to the state of tree in that cell.

    Parameters
    ----------
    img : NDArray
        image of wildfire grid with tree missing at agent location
    agent : Agent
        agent located in the tile to be re-rendered
    helper_grid : Grid
        grid containing only trees and no agents. Used to get the state of tree in the cell containing the agent
    world : WorldT
        wildfire world
    x_min : list[int], optional
        list of x-coordinates of the left boundary of selfish regions
    y_min : list[int], optional
        list of y-coordinates of the top boundary of selfish regions
    x_max : list[int], optional
        list of x-coordinates of the right boundary of selfish regions
    y_max : list[int], optional
        list of y-coordinates of the bottom boundary of selfish regions
    colors : list, optional
        list of colors of the boundaries of selfish regions. Boundary color is same as the color of the corresponding selfish agent

    Returns
    -------
    img : NDArray
        image with all trees and agents rendered
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
    i, j = pos
    changed_left_boundary = False
    changed_top_boundary = False
    if colors is not None:
        for index, color in enumerate(colors):
            # check if object is located adjacent to the top boundary of selfish region
            if j == y_min[index]:
                if x_min[index] <= i <= x_max[index]:
                    changed_top_boundary = True
                    fill_coords(
                        img[ymin:ymax, xmin:xmax, :],
                        point_in_rect(0, 1, 0, 0.093),
                        color,
                    )
            # check if object is located adjacent to the left boundary of selfish region
            if i == x_min[index]:
                if y_min[index] <= j <= y_max[index]:
                    changed_left_boundary = True
                    fill_coords(
                        img[ymin:ymax, xmin:xmax, :],
                        point_in_rect(0, 0.093, 0, 1),
                        color,
                    )
            # check if object is located adjacent to the bottom boundary of selfish region
            if j == y_max[index] + 1:
                if x_min[index] <= i <= x_max[index]:
                    changed_top_boundary = True
                    fill_coords(
                        img[ymin:ymax, xmin:xmax, :],
                        point_in_rect(0, 1, 0, 0.093),
                        color,
                    )
            # check if object is located adjacent to the right boundary of selfish region
            if i == x_max[index] + 1:
                if y_min[index] <= j <= y_max[index]:
                    changed_left_boundary = True
                    fill_coords(
                        img[ymin:ymax, xmin:xmax, :],
                        point_in_rect(0, 0.093, 0, 1),
                        color,
                    )
    # use default boundary color if cell is not on boundary of selfish region
    if not changed_left_boundary:
        fill_coords(
            img[ymin:ymax, xmin:xmax, :], point_in_rect(0, 0.031, 0, 1), (100, 100, 100)
        )
    if not changed_top_boundary:
        fill_coords(
            img[ymin:ymax, xmin:xmax, :], point_in_rect(0, 1, 0, 0.031), (100, 100, 100)
        )

    return img


def get_initial_fire_coordinates(x, y, grid_size, n, without_wall_coordinates=False):
    """Generate the coordinates of trees on fire in a uniformly randomly located square fire region of size n by n.

    Parameters:
    ----------
        x : int
            the x-coordinate of the center cell of the fire region if n is odd, or the x-coordinate of the top-left corner cell of the fire region if n is even
        y : int
            the y-coordinate of the center cell of the fire region if n is odd, or the y-coordinate of the top-left corner cell of the fire region if n is even
        grid_size : int
            the side of the square grid
        n : int
            the side of the square fire region
        without_wall_coordinates : bool, optional
            if True, the coordinates (x, y) are assumed to be given in, and output coordinates are returned in, grid without wall coordinates.

    Returns:
    -------
    coordinates : list(tuple(int, int))
        a list of tuples, where each tuple represents the position coordinates of a tree on fire in the fire region.
    """

    if without_wall_coordinates:
        # convert grid without wall coordinates to grid with wall coordinates
        x += 1
        y += 1

    if n % 2 == 0:
        # side of the fire region is an even number
        coordinates = []
        # loop through the positions of cells in the fire region. The top-left corner cell is (x, y)
        for i in range(x, x + n):
            for j in range(y, y + n):
                coordinates.append((i, j))
        return coordinates
    else:
        # side of the fire region is an odd number
        # offset is the distance from the center cell to the edge cell of a fire region of size n by n
        offset = int((n - 1) / 2)

        # determine range of x and y coordinates lying within the fire region. The center cell is (x, y).
        start_x = max(1, x - offset)
        end_x = min(grid_size, x + offset)
        start_y = max(1, y - offset)
        end_y = min(grid_size, y + offset)

        coordinates = []
        # loop through the positions of cells in the fire region
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                if without_wall_coordinates:
                    coordinates.append((x - 1, y - 1))
                else:
                    coordinates.append((x, y))
        return coordinates
