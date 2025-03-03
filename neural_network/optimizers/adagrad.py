import numpy as np
from ..common import ParamDict
from .optimizer import Optimizer

class AdaGrad(Optimizer):
    """
    Adaptive Gradient
    =====

    Parameters
    ----------
    learn_rate : float

    Math
    ----
    G_(0) = 0.0

    G_(t) = G_(t-1) + grad(w_(t))^2 \\
    LR_scaled = LR / sqrt(G_(t) + 1e-12)

    w_(t+1) = w_(t) - LR_scaled * grad(w_(t))
    """
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

        # accumulated squared gradients
        self.accu_sq_grad = {}
    
    def step(self, parameters: list[ParamDict]) -> None:

        for param in parameters:

            weight: np.ndarray = param['weight'].tensor

            param_id = id(param['weight'])
            if param_id not in self.accu_sq_grad:
                self.accu_sq_grad[param_id] = np.zeros_like(weight)

            # accumulate squared gradients
            self.accu_sq_grad[param_id] += np.square(param['grad'])

            # calculate new learn rate
            LR_scaled = np.full_like(weight, self.LR) / np.sqrt(self.accu_sq_grad[param_id] + 1e-12)

            # update weights
            weight += -1 * LR_scaled * param['grad']

            param['weight'].assign(weight)

        self._clip_params(parameters)
