from ..exceptions import InputValidationError
from ..print_utils import PrintUtils
from ..utils import Utils

class Layer:

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str,
                 l2_regularizer: float) -> None:

        if input_size == 0:
            raise InputValidationError("A layer can't have 0 input.")
        if output_size == 0:
            raise InputValidationError("A layer can't have 0 output (0 neurons).")
        
        # L2 Regularization Strength
        # low reg strength -> cook in class, and fail in exam; overfit
        # high reg strength -> I am dumb dumb, can't learn; underfit
        # Large weights and biases will are penalized more aggressively than small ones
        # Don't set it too large, at most 0.01 (unless you know what you're doing)
        #     regularized_loss     = parameter_los      + 1/2 * lambda_param * parameter^2
        #     regularized_gradient = parameter_gradient +       lambda_param * parameter
        if l2_regularizer < 0.0:
            raise InputValidationError("Regularization Strength can't be negative.")
        if l2_regularizer > 0.01:
            PrintUtils.print_warning(f"Warning: Regularization Strength {l2_regularizer:.3f} is strong. Consider keeping it less than 0.01")

        activation = activation.strip().lower()
        Utils._act_func_validator(activation)

        self.input_size = input_size
        self.output_size = output_size
        self.act_name = activation

        self.l2_lambda = l2_regularizer

    def build(self, is_first: bool, is_final: bool):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
    def optimize(self):
        raise NotImplementedError
    
    def _get_param_count(self):
        raise NotImplementedError
    
    @property
    def requires_training_flag(self):
        return False