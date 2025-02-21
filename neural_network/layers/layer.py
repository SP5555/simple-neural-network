from ..auto_diff.auto_diff_reverse import Tensor
from ..activations.activation import Activation
from ..exceptions import InputValidationError
from ..print_utils import PrintUtils

class Layer:
    """
    Abstract base class for all neural network layers.

    This class serves as a template and cannot be used directly in the models. \\
    Methods in this class raise `NotImplementedError` to enforce implementation 
    in derived child classes.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Activation,
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

        self.input_size = input_size
        self.output_size = output_size

        self.activation = activation
        # build parameter for learnable activations
        self.activation.build_parameters(output_size)

        self.L2_lambda = weight_decay

        self.tmp_batch_size: int = None

        # tensor/operation auto-diff objects
        self._A: Tensor = None
        self._out: Tensor = None

    def build(self, is_first: bool, is_final: bool):
        raise NotImplementedError

    def compile(self, A: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, is_training: bool = False):
        raise NotImplementedError

    # After one forward pass sweep, this backward call is called
    # only once on the last layer
    def backward(self, seed: Tensor):
        
        # Math
        # Z = W*A_in + B
        # A = activation(Z, learn_b) # learnable parem is used only in some activations

        # derivative of loss w.r.t. weights
        # dL/dW
        # = dL/dA * dA/dZ * dZ/dW
        # = dL/dA * dA/dZ * A_in

        # derivative of loss w.r.t. biases
        # dL/db(n)
        # = dL/dA * dA/dZ * dZ/dB
        # = dL/dA * dA/dZ * 1

        # derivative of loss w.r.t. learnable parameter (if exists)
        # dL/dlearn_b
        # = dL/dA * dA/dlearn_b

        # "seed" or gradient of loss for previous layer
        # NOTE: A_in affects all A, so backpropagation to A_in will be related to all A
        # dL/dA_in
        # = dL/dA * dA/dZ * dZ/dA_in
        # = dL/dA * dA/dZ * W

        # but auto-diff did all that with this single call. No headaches LOL
        self.activation.backward(seed)

    def regularize_grads(self):
        raise NotImplementedError

    def _get_weights_and_grads(self) -> list[dict]:
        raise NotImplementedError

    def zero_grads(self):
        raise NotImplementedError

    def _get_param_count(self) -> int:
        raise NotImplementedError
