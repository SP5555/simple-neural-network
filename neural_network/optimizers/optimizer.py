from ..exceptions import InputValidationError
from ..print_utils import PrintUtils

class Optimizer:
    def __init__(self, learn_rate: float):

        # Learn Rate
        # How fast or slow this network learns
        #     new_parameter = old_parameter - velocity * learn_rate
        if learn_rate <= 0.0:
            raise InputValidationError("Learn rate must be positive.")
        if learn_rate >= 0.1:
            PrintUtils.print_warning(f"Warning: Learn rate {learn_rate:.3f} may cause instability. Consider keeping it less than 0.1.")

        self.LR = learn_rate

    def step(self, parameters: list) -> None:
        raise NotImplementedError