import numpy as np
from ..common import PrintUtils, ParamDict, InputValidationError
from .optimizer import Optimizer

class RMSprop(Optimizer):
    """
    Root Mean Square Propagation
    =====

    Parameters
    ----------
    learn_rate : float
    
    avg_decay_rate : float

    Math
    ----
    v_(0) = 0.0

    v_(t) = momentum * v_(t-1) + (1 - momentum) * grad(w_(t))^2 \\
    LR_scaled = LR / sqrt(v_(t) + 1e-12)

    w_(t+1) = w_(t) - LR_scaled * grad(w_(t))
    """
    def __init__(self, learn_rate: float, avg_decay_rate: float = 0.8):
        super().__init__(learn_rate)

        # Moving Average Rate for "exponential moving average"
        # conceptually similar to momentum
        # 0.0 disables the momentum behavior
        # 1.0 disables updates
        # MUST be less than 1.0
        if avg_decay_rate < 0.0:
            raise InputValidationError("Average gradient decay rate can't be negative.")
        if avg_decay_rate >= 1.0:
            raise InputValidationError("Average gradient decay rate must be less than 1.0.")
        if avg_decay_rate >= 0.95:
            PrintUtils.print_warning(f"Warning: avg_decay_rate = {avg_decay_rate:.3f} may cause strong \"gliding\" behavior. " +
                                     " Consider keeping it less than 0.95")

        self.mom = avg_decay_rate
        # moving averages
        self.mov_avg = {}
    
    def step(self, parameters: list[ParamDict]):

        for param in parameters:

            weight: np.ndarray = param['weight'].tensor

            param_id = id(param['weight'])
            if param_id not in self.mov_avg:
                self.mov_avg[param_id] = np.zeros_like(weight)

            # update moving average
            self.mov_avg[param_id] = self.mom * self.mov_avg[param_id] + (1 - self.mom) * np.square(param['grad'])

            LR_scaled = np.full_like(weight, self.LR) / np.sqrt(self.mov_avg[param_id] + 1e-12)
            
            # update weights
            weight += -1 * LR_scaled * param['grad']
            
            param['weight'].assign(weight)

        self._clip_params(parameters)