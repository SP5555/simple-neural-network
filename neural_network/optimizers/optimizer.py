import numpy as np
from ..common import PrintUtils, ParamDict, InputValidationError

class Optimizer:
    """
    Abstract base class for all optimizers.

    This class serves as a template and cannot be used directly in the models. \\
    Methods in this class raise `NotImplementedError` to enforce implementation 
    in derived child classes.
    """
    def __init__(self, learn_rate: float):

        # Learn Rate
        # How fast or slow this network learns
        if learn_rate <= 0.0:
            raise InputValidationError("Learn rate must be positive.")
        if learn_rate >= 0.1:
            PrintUtils.print_warning(f"Warning: learn_rate = {learn_rate:.3f} may cause instability. Consider keeping it less than 0.1.")

        self.LR = learn_rate

    def step(self, parameters: list[ParamDict]) -> None:
        # parameter is a list of dictionaries
        # Keys: 'weight', 'grad'
        raise NotImplementedError

    def _clip_params(self, parameters: list[ParamDict]) -> None:
        """Clips learnable parameters."""
        for param in parameters:
            if 'learnable' in param:
                low, high = param['constraints']
                weight = param['weight'].tensor
                param['weight'].assign(np.clip(weight, low, high))