import numpy as np
from ..auto_diff.auto_diff_reverse import Tensor
from ..common import ParamDict

class Layer:
    """
    Abstract base class for all neural network layers.

    This class serves as a template and cannot be used directly in the models. \\
    Methods in this class raise `NotImplementedError` to enforce implementation 
    in derived child classes.
    """
    def __init__(self) -> None:
        
        self.input_size = None
        self.neuron_count = None

        self._tmp_batch_size: int = None

        # tensor/operation auto-diff objects
        self._out: Tensor = None

    def build(self, A: Tensor, input_size: int) -> tuple[Tensor, int]:
        """
        Constructs the computation graph for this layer.

        This method defines how the layer connects to the preceding tensor(s) and
        determines the output shape. It must be implemented by subclasses.

        Parameters
        ----------
        A : Tensor
            The input tensor that this layer depends on when constructing the computation graph.

        input_size : int
            The number of input features expected by this layer.

        Returns
        -------
        tuple[Tensor, int]
            A tuple containing the output tensor and its size.
        """
        raise NotImplementedError

    def pre_setup_tensors(self, batch_size: int, is_training: bool = False):
        """
        Prepares necessary internal tensors **before** each forward pass.

        Some layers, such as Dropout, require updating internal state (e.g., sampling dropout masks)
        before computing activations. This function ensures such adjustments are made
        before executing forward and backward passes.

        Parameters
        ----------
        batch_size : int
            Number of samples in a single forward pass.

        is_training : bool, optional
            Whether the model is in training mode. Certain layers (e.g., Dropout) 
            behave differently during training vs. inference. Default is `False`.
        """
        raise NotImplementedError

    def post_setup_tensors(self, is_training: bool = False):
        """
        Updates internal tensors **after** a forward-backward pass.

        Some layers, such as BatchNorm, update running averages (e.g., running mean and variance)
        after each backward pass. This function ensures such statistics are updated
        before the next iteration.

        Parameters
        ----------
        is_training : bool, optional
            Whether the model is in training mode. Some layers update statistics 
            (e.g., BatchNorm's running averages) only during training. Default is `False`.
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
        # Math (for dense layer)
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

    # ===== gradient-related utility functions =====
    def prepare_grads(self):
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
