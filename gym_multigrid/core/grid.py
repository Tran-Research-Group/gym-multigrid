# pylint: disable=line-too-long, dangerous-default-value, unused-wildcard-import, wildcard-import
from typing import Type
from copy import deepcopy
import numpy as np
from gym_multigrid.core.world import WorldT
from gym_multigrid.utils.rendering import *
from gym_multigrid.core.object import WorldObj, Wall, WorldObjT
from gym_multigrid.core.constants import TILE_PIXELS


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width: int, height: int, world: WorldT):
        """Create a grid of a given width and height in given world

        Parameters
        ----------
        width : int
            width of the grid
        height : int
            height of the grid
        world : WorldT
            world object in which the grid is situated
        """
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        self.world = world

        self.grid: list[WorldObjT | None] = [None] * width * height

    def __contains__(self, key: type[WorldObjT] | tuple) -> bool:
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other: "Grid") -> bool:
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other: "Grid") -> bool:
        return not self == other

    def copy(self) -> "Grid":
        """Create a deep copy of the grid

        Returns
        -------
        Grid
            deep copy of the grid
        """

        return deepcopy(self)

    def set(self, i: int, j: int, v: WorldObjT | None) -> None:
        """Insert the given object at the given position in the grid

        Parameters
        ----------
        i : int
            x-coordinate of the position
        j : int
            y-coordinate of the position
        v : WorldObjT | None
            object to be inserted
        """
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i: int, j: int) -> WorldObjT | None:
        """Get the object at the given position in the grid

        Parameters
        ----------
        i : int
            x-coordinate of the position
        j : int
            y-coordinate of the position

        Returns
        -------
        WorldObjT | None
            object at the given position in the grid
        """
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Type[WorldObjT] = Wall,
    ) -> None:
        """Create a horizontal wall starting from given point (x, y) and of given length.

        Parameters
        ----------
        x : int
            x-coordinate of the starting point
        y : int
            y-coordinate of the starting point
        length : int | None, optional
            length of the wall, by default None
        obj_type : Type[WorldObjT], optional
            type of object to be inserted, by default Wall
        """
        if length is None:
            length = self.width - x
        assert length is not None
        for i in range(0, length):
            wall_obj = obj_type(self.world)
            wall_obj.pos = (x + i, y)
            self.set(x + i, y, wall_obj)

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Type[WorldObjT] = Wall,
    ):
        """Create a vertical wall starting from given point (x, y) and of given length.

        Parameters
        ----------
        x : int
            x-coordinate of the starting point
        y : int
            y-coordinate of the starting point
        length : int | None, optional
            length of the wall, by default None
        obj_type : Type[WorldObjT], optional
            type of object to be inserted, by default Wall
        """
        if length is None:
            length = self.height - y
        for j in range(0, length):
            wall_obj = obj_type(self.world)
            wall_obj.pos = (x, y + j)
            self.set(x, y + j, wall_obj)

    def wall_rect(self, x: int, y: int, w: int, h: int) -> None:
        """Create a rectangle of walls starting from given point (x, y) and of given width and height.

        Parameters
        ----------
        x : int
            x-coordinate of the starting point
        y : int
            y-coordinate of the starting point
        w : int
            width of the rectangle
        h : int
            height of the rectangle
        """
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self) -> "Grid":
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width, self.world)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid. The subset is a rectangle of size width x height whose top-left corner is at (topX, topY).

        Parameters
        ----------
        topX : int
            x-coordinate of the top-left corner of the subset of grid
        topY : int
            y-coordinate of the top-left corner of the subset of grid
        width : int
            width of the subset of grid
        height : int
            height of the subset of grid
        """

        grid = Grid(width, height, self.world)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(self.world)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        world: WorldT,
        obj: WorldObjT | None,
        highlights: list[bool] = None,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
        cache: bool = True,
        cell_location: int = 0,
        selfish_boundary_color: tuple[int, int, int] = (100, 100, 100),
    ):
        """
        Render a tile and cache the result

        Parameters
        ----------
        world : WorldT
            world object in which the grid is situated
        obj : WorldObjT | None
            object to be rendered
        highlights : list[bool], optional
            list of booleans indicating whether to highlight the tile, by default None
        tile_size : int, optional
            size of the tile, by default TILE_PIXELS
        subdivs : int, optional
            number of subdivisions to use for downsampling image, by default 3
        cache : bool, optional
            whether to cache the rendered tile, by default True
        cell_location : int, optional
            determine if the cell is located adjacent to a selfish region boundary, by default 0. Only applicable for wildfire environment.
        selfish_boundary_color : tuple[int, int, int], optional
            color of the selfish region boundary, by default (100, 100, 100). Only applicable for wildfire environment.

        """
        # Key for caching
        key = (*highlights, tile_size)
        key = obj.encode() + key if obj else key
        if cell_location != 0:
            key = (key, (cell_location, selfish_boundary_color.tobytes()))

        # Return the cached tile if it exists
        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # render the object
        if obj is not None:
            obj.render(img)

        # create grid lines around object (specifically the top and left boundaries for each cell)
        changed_left_boundary = False
        changed_top_boundary = False
        if cell_location == 1 or cell_location == 3:
            changed_top_boundary = True
            fill_coords(img, point_in_rect(0, 1, 0, 0.093), selfish_boundary_color)
        if cell_location == 2 or cell_location == 3:
            changed_left_boundary = True
            fill_coords(img, point_in_rect(0, 0.093, 0, 1), selfish_boundary_color)

        # use default boundary color if cell is not on boundary of selfish region
        if not changed_left_boundary:
            fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        if not changed_top_boundary:
            fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # Highlight the cell  if needed
        if len(highlights) > 0:
            for h in highlights:
                highlight_img(
                    img,
                    color=world.COLORS[world.IDX_TO_COLOR[h % len(world.IDX_TO_COLOR)]],
                )

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        if cache:
            cls.tile_cache[key] = img
        else:
            pass

        return img

    def render(
        self,
        tile_size,
        highlight_masks=None,
        uncached_object_types: list[str] = None,
        x_min: list[int] = None,
        y_min: list[int] = None,
        x_max: list[int] = None,
        y_max: list[int] = None,
        colors: list[tuple[int, int, int]] = None,
    ):
        """
        Render this grid at a given scale

        Parameters
        ----------
        tile_size : int
            size of the tile
        highlight_masks : list[bool], optional
            list of booleans indicating whether to highlight the tile, by default None
        uncached_object_types : list[str], optional
            list of object types that should not be cached, by default None
        x_min : list[int], optional
            list of x-coordinates of the left boundary of selfish regions, by default None. Only applicable for wildfire environment.
        y_min : list[int], optional
            list of y-coordinates of the top boundary of selfish regions, by default None. Only applicable for wildfire environment.
        x_max : list[int], optional
            list of x-coordinates of the right boundary of selfish regions, by default None. Only applicable for wildfire environment.
        y_max : list[int], optional
            list of y-coordinates of the bottom boundary of selfish regions, by default None. Only applicable for wildfire environment.
        colors : list[tuple[int,int,int]], optional
            list of colors to use for selfish region boundaries, by default None. Only applicable for wildfire environment.
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                cache: bool = True
                if cell is not None and cell.type in uncached_object_types:
                    cache = False
                if x_min is not None:
                    # determine if the cell is located adjacent to a selfish region boundary
                    cell_location = 0
                    selfish_boundary_color = (100, 100, 100)
                    for index, color in enumerate(colors):
                        # check if object is located adjacent to the top boundary of selfish region
                        if j == y_min[index]:
                            if x_min[index] <= i <= x_max[index]:
                                cell_location = 1
                                selfish_boundary_color = color
                        # check if object is located adjacent to the left boundary of selfish region
                        if i == x_min[index]:
                            if y_min[index] <= j <= y_max[index]:
                                cell_location += 2
                                selfish_boundary_color = color
                        # check if object is located adjacent to the bottom boundary of selfish region
                        if j == y_max[index] + 1:
                            if x_min[index] <= i <= x_max[index]:
                                cell_location = 1
                                selfish_boundary_color = color
                        # check if object is located adjacent to the right boundary of selfish region
                        if i == x_max[index] + 1:
                            if y_min[index] <= j <= y_max[index]:
                                cell_location = 2
                                selfish_boundary_color = color
                    # render the tile
                    tile_img = Grid.render_tile(
                        self.world,
                        cell,
                        highlights=(
                            [] if highlight_masks is None else highlight_masks[i, j]
                        ),
                        tile_size=tile_size,
                        cache=cache,
                        cell_location=cell_location,
                        selfish_boundary_color=selfish_boundary_color,
                    )
                else:
                    # render the tile without selfish region boundary
                    tile_img = Grid.render_tile(
                        self.world,
                        cell,
                        highlights=(
                            [] if highlight_masks is None else highlight_masks[i, j]
                        ),
                        tile_size=tile_size,
                        cache=cache,
                    )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask: np.ndarray[bool] | None = None) -> np.ndarray:
        """
        Produce a compact numpy encoding of the grid

        Parameters
        ----------
        vis_mask : np.ndarray[bool] | None, optional
            mask specifying visible regions of grid, by default None

        Returns
        -------
        np.ndarray
            compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros(
            (self.width, self.height, self.world.encode_dim), dtype="uint8"
        )

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = self.world.OBJECT_TO_IDX["empty"]
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if self.world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(self.world)

        return array

    def encode_for_agents(
        self, agent_pos: tuple[int, int], vis_mask: np.ndarray[bool] | None = None
    ) -> np.ndarray:
        """
        Produce a compact numpy encoding of the grid

        Parameters
        ----------
        agent_pos : tuple[int, int]
            position of the agent
        vis_mask : np.ndarray[bool] | None, optional
            mask specifying visible regions of grid, by default None

        Returns
        -------
        np.ndarray
            compact numpy encoding of the grid
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros(
            (self.width, self.height, self.world.encode_dim), dtype="uint8"
        )

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = self.world.OBJECT_TO_IDX["empty"]
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if self.world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(
                            current_agent=np.array_equal(agent_pos, (i, j))
                        )

        return array

    def process_vis(self, grid: "Grid", agent_pos: tuple[int, int]) -> np.ndarray[bool]:
        """Returns a mask of the visible cells in the grid

        Parameters
        ----------
        grid : Grid

        agent_pos : tuple[int, int]
            position of the agent

        Returns
        -------
        np.ndarray[bool]
            mask of the visible cells in the grid
        """
        mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width - 1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask
