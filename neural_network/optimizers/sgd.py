import numpy as np
from ..common import ParamDict
from .optimizer import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    =====

    Parameters
    ----------
    learn_rate : float

    Math
    ----
    w_(t+1) = w_(t) - LR * grad(w_(t))
    """
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

    def step(self, parameters: list[ParamDict]) -> None:

        for param in parameters:

            weight: np.ndarray = param['weight'].tensor

            # update weights
            weight += -1 * self.LR * param['grad'] 

            param['weight'].assign(weight)

        self._clip_params(parameters)