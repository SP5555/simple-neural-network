import numpy as np
from ..auto_diff.auto_diff_reverse import Tensor
from ..activations.activation import Activation
from ..common import ParamDict
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
        self.activation.build_alpha_tensor(output_size)

        self.L2_lambda = weight_decay

        self.tmp_batch_size: int = None

        # tensor/operation auto-diff objects
        self._out: Tensor = None

    def build(self, is_first: bool, is_final: bool):
        """
        Initializes the internal tensors (weights and biases) for the layer.

        Parameters
        ----------
        is_first : bool
            Indicates if this is the first layer in the model. Default is `False`.
        
        is_final : bool
            Indicates if this is the final layer in the model. Default is `False`.
        """
        raise NotImplementedError

    def compile(self, A: Tensor) -> Tensor:
        """
        Constructs the computation graph using the initialized internal tensors.

        This method defines the mathematical operations of the layer.

        Parameters
        ----------
        A : Tensor
            The input tensor into this layer.

        Returns
        -------
        Tensor
            The output tensor after applying the layer's transformation and activation.
        """
        raise NotImplementedError

    def setup_tensors(self, batch_size: int, is_training: bool = False):
        """
        Updates internal tensors that depend on batch size and training mode.

        Some layers, such as Dropout, require dynamically adjusting internal
        tensors (e.g., masks) before forward and backward passes. This method
        ensures such tensors are properly configured based on the given batch size
        and whether the model is in training mode.

        Parameters
        ----------
        batch_size : int
            Number of samples in a single forward pass.

        is_training : bool, optional
            Whether the model is in training mode. Affects certain layers
            (e.g., Dropout) that behave differently during training and inference.
            Default is `False`.
        """
        raise NotImplementedError

    def forward(self):
        """
        Computes all tensors in the computation graph up to the point
        where this forward pass is called.

        This function should be called only once at the final layer
        during inference or at the loss "layer" during training.
        """
        self._out.forward()

    def backward(self, seed: Tensor):
        """
        Performs the backward pass, computing gradients for all tensors in the
        computation graph based on the provided seed (gradient of the loss).

        This function should be called only once, at the last layer of the network,
        to propagate gradients back through the entire model.

        The mathematical foundation follows standard backpropagation:
        - Computes gradients for weights, biases, and learnable activation parameters (if any).
        - Uses the chain rule to propagate gradients from the loss function.
        - Automatically computes and tracks required gradients via auto-differentiation.

        Auto-diff handles the gradient computations efficiently,
        eliminating the need for manual differentiation.
        """
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
        self._out.backward(seed)
    
    def evaluate(self) -> np.ndarray:
        return self._out.evaluate()

    def regularize_grads(self):
        """
        Applies L2 regularization to all computed gradients.
        """
        raise NotImplementedError


    def _get_weights_and_grads(self) -> list[ParamDict]:
        raise NotImplementedError

    def zero_grads(self):
        """
        Resets the accumulated gradient (partial) of
        all tensors in this layer to zero.

        This is necessary before performing a new backward pass,
        as gradients are accumulated in reverse-mode auto-diff.
        If not cleared, calling backward() multiple times will 
        result in accumulated gradients from previous passes.
        """
        raise NotImplementedError

    def _get_param_count(self) -> int:
        raise NotImplementedError
