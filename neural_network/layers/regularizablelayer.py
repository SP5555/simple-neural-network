from .layer import Layer
from ..auto_diff.auto_diff_reverse import Tensor
from ..common import PrintUtils, InputValidationError

class RegularizableLayer(Layer):
    """
    Abstract base class for layers that support weight regularization.

    Inherits from `Layer` and adds support for L2 weight decay through the 
    `weight_decay` parameter. This class is intended for layers with learnable 
    parameters (e.g., weights, scaling factors) that benefit from regularization. \\
    It should not be instantiated directly, only used as a parent for layers 
    like `Dense` or `BatchNorm` that require regularization logic.
    
    Parameters
    ----------
    weight_decay : float, optional
        L2 regularization strength for the weights.
        If not defined, it will use the global `weight_decay` value 
        defined in the `NeuralNetwork` class level.
    """
    def __init__(self, weight_decay: float | None):

        # L2 Regularization Strength
        # low reg strength -> cook in class, and fail in exam; overfit
        # high reg strength -> I am dumb dumb, can't learn; underfit
        # Large weights and biases will are penalized more aggressively than small ones
        # Don't set it too large, at most 0.01 (unless you know what you're doing)
        #     regularized_loss     = parameter_loss     + 1/2 * L2_lambda * parameter^2
        #     regularized_gradient = parameter_gradient +       L2_lambda * parameter
        if weight_decay is not None:
            if weight_decay < 0.0:
                raise InputValidationError("Regularization Strength can't be negative.")
            if weight_decay > 0.01:
                PrintUtils.print_warning(f"[{self.__class__.__name__}] Warning: " + 
                                        f"Regularization Strength {weight_decay:.3f} is strong. " +
                                        "Consider keeping it less than 0.01")

        super().__init__()

        self.weight_decay = weight_decay

    def build(self, A: Tensor, input_size: int, weight_decay: float | None) -> tuple[Tensor, int]:
        raise NotImplementedError
