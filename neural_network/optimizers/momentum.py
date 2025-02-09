import numpy as np
from ..exceptions import InputValidationError
from ..print_utils import PrintUtils
from .optimizer import Optimizer

class Momentum(Optimizer):
    """
    Momentum

    v_(0) = 0.0

    v_(t) = momentum * v_(t-1) + (1 - momentum) * grad(w_(t))

    w_(t+1) = w_(t) - LR * v_(t)
    """
    def __init__(self, learn_rate: float, momentum: float = 0.8):
        super().__init__(learn_rate)

        # Momentum for Momentum Gradient Descent
        # 0.0 disables the momentum behavior
        # having momentum helps escape the high loss "plateaus" better
        # high values result in smoother/stronger "gliding" descent
        # MUST be less than 1.0
        if momentum < 0.0:
            raise InputValidationError("Momentum can't be negative.")
        if momentum >= 1.0:
            raise InputValidationError("Momentum must be less than 1.0.")
        if momentum >= 0.95:
            PrintUtils.print_warning(f"Warning: momentum = {momentum:.3f} may cause strong \"gliding\" behavior. " +
                                     "Consider keeping it less than 0.95")

        self.mom = momentum
        # velocities
        self.v = {}

    def step(self, parameters: list[dict]) -> None:

        for param in parameters:

            param_id = id(param['weight'])
            if param_id not in self.v:
                self.v[param_id] = np.zeros_like(param['weight'])

            # update velocity
            self.v[param_id] = self.mom * self.v[param_id] + (1 - self.mom) * param['grad']
            
            # update weights
            param['weight'] += -1 * self.LR * self.v[param_id]

        self._clip_params(parameters)