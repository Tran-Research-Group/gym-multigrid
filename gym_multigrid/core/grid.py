# pylint: disable=line-too-long, dangerous-default-value, unused-wildcard-import, wildcard-import
from copy import deepcopy
from typing import Type

import numpy as np
from gym_multigrid.core.constants import TILE_PIXELS
from gym_multigrid.core.object import Wall, WorldObj
from gym_multigrid.core.world import World
from gym_multigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
)
from numpy.typing import NDArray


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache: dict[tuple, NDArray[np.uint8]] = {}

    def __init__(self, width: int, height: int, world: World):
        """Create a grid of a given width and height in given world

        Parameters
        ----------
        width : int
            width of the grid
        height : int
            height of the grid
        world : World
            world object in which the grid is situated
        """
        # assert width > 0
        # assert height > 0

        self.width: int = width
        self.height: int = height
        self.world: World = world

        if self.world.encode_dim == 3:
            self.empty_encoding = np.array([self.world.OBJECT_TO_IDX["empty"], 0, 0])
        elif self.world.encode_dim == 6:
            self.empty_encoding = np.array(
                [self.world.OBJECT_TO_IDX["empty"], 0, 0, 0, 0, 0]
            )

        self.grid: list[WorldObj | None] = [None for _ in range(width * height)]

    def __contains__(self, key: WorldObj | tuple) -> bool:
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

    def __eq__(self, other: object) -> bool:
        grid1 = self.encode()
        if not isinstance(other, Grid):
            return False
        else:
            grid2 = other.encode()
            return np.array_equal(grid2, grid1)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def copy(self) -> "Grid":
        """Create a deep copy of the grid

        Returns
        -------
        Grid
            deep copy of the grid
        """

        return deepcopy(self)

    def set(self, i: int, j: int, v: WorldObj | None) -> None:
        """Insert the given object at the given position in the grid

        Parameters
        ----------
        i : int
            x-coordinate of the position
        j : int
            y-coordinate of the position
        v : WorldObj | None
            object to be inserted
        """
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i: int, j: int) -> WorldObj | None:
        """Get the object at the given position in the grid

        Parameters
        ----------
        i : int
            x-coordinate of the position
        j : int
            y-coordinate of the position

        Returns
        -------
        WorldObj | None
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
        obj_type: Type[WorldObj] = Wall,
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
        obj_type : Type[WorldObj], optional
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
        obj_type: Type[WorldObj] = Wall,
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
        obj_type : Type[WorldObj], optional
            type of object to be inserted, by default Wall
        """
        if length is None:
            length = self.height - y
        for j in range(0, length):
            wall_obj = obj_type(self.world)
            wall_obj.pos = (x, y + j)
            self.set(x, y + j, wall_obj)

    def rect_filled(self, x: int, y: int, w: int, h: int, obj: WorldObj) -> None:
        for i in range(w):
            for j in range(h):
                self.set(x + i, y + j, obj)

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

    def wall_rect_filled(self, x: int, y: int, w: int, h: int) -> None:
        for i in range(w):
            for j in range(h):
                self.set(x + i, y + j, Wall(self.world))

    def rotate_left(self) -> "Grid":
        """
        Rotate the grid to the left (counter-clockwise)
        """

        # TODO pretty inefficient, creates + populates an entirely new grid object each time this method is called
        # could just apply a transformation matrix to an existing grid object or whatever
        grid = Grid(self.height, self.width, self.world)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, top_x: int, top_y: int, width: int, height: int):
        """
        Get a subset of the grid. The subset is a rectangle of size width x height whose top-left corner is at (topX, topY).

        Parameters
        ----------
        top_x : int
            x-coordinate of the top-left corner of the subset of grid
        top_y : int
            y-coordinate of the top-left corner of the subset of grid
        width : int
            width of the subset of grid
        height : int
            height of the subset of grid
        """

        # TODO pretty inefficient, creates + populates an entirely new grid object each time this method is called
        grid = Grid(width, height, self.world)

        for j in range(0, height):
            for i in range(0, width):
                x = top_x + i
                y = top_y + j

                if x >= 0 and x < self.width and y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(self.world)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        world: World,
        obj: WorldObj | None,
        highlights: list[bool] = [],
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
        world : World
            world object in which the grid is situated
        obj : WorldObj | None
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

        if obj is not None:
            # get object index if it has one to differentiate objects with same encoding
            obj_index = (obj.index,) if hasattr(obj, "index") else ()
            assigned_agent_index = (
                (obj.assigned_agent_index,)
                if hasattr(obj, "assigned_agent_index")
                else ()
            )
            key = key + obj.encode() + obj_index + assigned_agent_index

        if cell_location != 0:
            key = (key, (cell_location, np.array(selfish_boundary_color).tobytes()))

        # Return the cached tile if it exists
        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img: NDArray[np.uint8] = np.zeros(
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
            # brighten the for each agent that sees it
            for h in highlights:
                highlight_img(img, color=world.COLORS["white"])

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
        uncached_object_types: list[str] = [],
        x_min: list[int] = [],
        y_min: list[int] = [],
        x_max: list[int] = [],
        y_max: list[int] = [],
        colors: list[tuple[int, int, int]] = [],
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

    def encode(self, vis_mask: NDArray[np.bool_] | None = None) -> np.ndarray:
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
                        array[i, j, :] = self.empty_encoding
                    else:
                        array[i, j, :] = v.encode()

        return array

    def encode_for_agents(
        self,
        agent_pos: tuple[int, int],
        vis_mask: NDArray[np.bool_] | None = None,
        observe_other_agents: bool = False,
        fruit_obs_mask: dict | None = None,
        agent=None,
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
            vis_mask = np.ones((self.width, self.height), dtype=np.bool)

        array = np.zeros(
            (self.width, self.height, self.world.encode_dim), dtype="uint8"
        )

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        array[i, j, :] = self.empty_encoding
                    # TODO update this here now that you have the mask working as expected
                    elif (
                        v.type == "fruit"
                        and fruit_obs_mask is not None
                        and not fruit_obs_mask[(agent, (i, j))]
                    ):
                        # fruit obs mask may prevent the agent from observing the fruit
                        array[i, j, :] = self.empty_encoding
                    elif (
                        v.type == "agent"
                        and (not np.array_equal(agent_pos, (i, j)))
                        and (not observe_other_agents)
                    ):
                        # agent cannot observe other agents
                        array[i, j, :] = self.empty_encoding
                    else:
                        # agent can observe itself
                        array[i, j, :] = v.encode(
                            current_agent=np.array_equal(agent_pos, (i, j))
                        )

        return array

    def process_vis(self, agent_pos: tuple[int, int]) -> NDArray[np.bool_]:
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
        mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, self.height)):
            for i in range(0, self.width - 1):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, self.width)):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, self.height):
            for i in range(0, self.width):
                if not mask[i, j]:
                    self.set(i, j, None)

        return mask
