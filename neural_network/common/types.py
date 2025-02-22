from typing import TypedDict
import numpy as np
from ..auto_diff.auto_diff_reverse import Tensor

class ParamDict(TypedDict, total=False):  
    weight: Tensor
    grad: np.ndarray
    learnable: bool
    constraints: tuple[float, float]
