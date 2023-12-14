import numpy as np

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([234, 153, 153]),
    "light_blue": np.array([90, 170, 223]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX: "dict[str, int]" = {key: i for i, key in enumerate(COLORS.keys())}

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

# Map of state names to integers
STATE_TO_IDX_WILDFIRE = {
    "healthy": 0,
    "on fire": 1,
    "burnt": 2,
}

# Map of state idx to color
STATE_IDX_TO_COLOR_WILDFIRE = {
    0: "green",
    1: "orange",
    2: "brown",
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

# Map of object types to short string
OBJECT_TO_STR = {
    "wall": "x",
    "floor": "F",
    "door": "D",
    "key": "K",
    "ball": "o",
    "box": "B",
    "goal": "G",
    "lava": "V",
    "agent": "a",
    "tree": "T",
}

# Short string for opened door
OPENED_DOOR_IDS = "_"

# Map agent's direction to short string
AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
