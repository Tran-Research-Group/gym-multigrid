from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Position: TypeAlias = tuple[int, int] | NDArray[np.int_]
Size: TypeAlias = tuple[int, int] | int
