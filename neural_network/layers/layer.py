from ..exceptions import InputValidationError
from ..print_utils import PrintUtils
from ..utils import Utils

class Layer:
    """
    Abstract base class for all neural network layers.

    This class serves as a template and cannot be used directly in the models.
    Methods in this class raise `NotImplementedError` to enforce implementation 
    in derived child classes.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str,
                 weight_decay: float) -> None:

        if input_size == 0:
            raise InputValidationError("A layer can't have 0 input.")
        if output_size == 0:
            raise InputValidationError("A layer can't have 0 output (0 neurons).")
        
        # L2 Regularization Strength
        # low reg strength -> cook in class, and fail in exam; overfit
        # high reg strength -> I am dumb dumb, can't learn; underfit
        # Large weights and biases will are penalized more aggressively than small ones
        # Don't set it too large, at most 0.01 (unless you know what you're doing)
        #     regularized_loss     = parameter_los      + 1/2 * L2_lambda * parameter^2
        #     regularized_gradient = parameter_gradient +       L2_lambda * parameter
        if weight_decay < 0.0:
            raise InputValidationError("Regularization Strength can't be negative.")
        if weight_decay > 0.01:
            PrintUtils.print_warning(f"Warning: Regularization Strength {weight_decay:.3f} is strong. Consider keeping it less than 0.01")

        activation = activation.strip().lower()
        Utils._act_func_validator(activation)

        self.input_size = input_size
        self.output_size = output_size
        self.act_name = activation

        self.L2_lambda = weight_decay

    def build(self, is_first: bool, is_final: bool):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
    def _get_params(self) -> list[dict]:
        raise NotImplementedError

    def _get_param_count(self) -> int:
        raise NotImplementedError
    
    @property
    def requires_training_flag(self):
        return False