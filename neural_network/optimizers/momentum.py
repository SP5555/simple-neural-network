import numpy as np
from ..exceptions import InputValidationError
from ..print_utils import PrintUtils
from .optimizer import Optimizer

class Momentum(Optimizer):
    def __init__(self, learn_rate: float, momentum: float):
        super().__init__(learn_rate)

        # Momentum for Momentum Gradient Descent
        # 0.0 disables the momentum behavior
        # having momentum helps escape the high loss "plateaus" better
        # high values result in smoother/stronger "gliding" descent
        # MUST be less than 1.0
        #     new_velocity = momentum * old_velocity + (1-momentum) * parameter_gradient
        if momentum < 0.0:
            raise InputValidationError("Momentum can't be negative.")
        if momentum >= 1.0:
            raise InputValidationError("Momentum must be less than 1.0.")
        if momentum >= 0.9:
            PrintUtils.print_warning(f"Warning: Momentum value {momentum:.3f} may cause strong \"gliding\" behavior. Consider keeping it less than 0.95")

        self.mom = momentum
        # velocities
        self.v = {}

    def step(self, parameters: list[dict]) -> None:
        # parameter is a list of dictionaries
        # Keys: 'weight', 'grad'
        for param in parameters:

            param_id = id(param['weight'])
            if param_id not in self.v:
                self.v[param_id] = np.zeros_like(param['weight'])

            # UPDATE velocity
            self.v[param_id] = self.mom * self.v[param_id] + (1 - self.mom) * param['grad']
            
            # UPDATE weights
            param['weight'] += -1 * self.v[param_id] * self.LR