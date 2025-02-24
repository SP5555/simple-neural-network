import numpy as np
from .auto_diff.auto_diff_reverse import Tensor
from .common import Metrics, Utils, PrintUtils, requires_build, InputValidationError
from .layers.layer import Layer

class NeuralNetwork:
    """
    Simple Neural Network
    =====
    A basic implementation of a feedforward neural network
    with support for multiple layer types, optimization methods, and loss functions. 
    It can be easily extended for various tasks such as classification and regression.

    Parameters
    ----------
    layers : list[Layer]
        List of supported layer classes. Minimum of one layer required.
    """
    def __init__(self,
                 layers: list[Layer]) -> None:
        
        self.utils = Utils(self)
        self.metrics = Metrics(self)
        
        # ===== ===== INPUT VALIDATION START ===== =====
        if not layers: # if list is empty
            raise InputValidationError("Empty layer configuration not possible.")
        # ===== ===== INPUT VALIDATION END ===== =====
        
        self._layers: list[Layer] = layers
        self._layer_count: int = len(layers)

        PrintUtils.print_info(f"[{self.__class__.__name__}] Neural network initialized.")

        self._is_built = False
    
    def build(self, input_size: int):
        """
        Builds necessary internal tensors and compiles the **computation graph** for the neural network.

        This function ensures that all layers are connected properly and constructs 
        the forward computation path only once, improving efficiency during training 
        and inference by avoiding redundant graph re-construction.

        Notes:
        - This function **must** be called before training or inference.
        - Any attempt to train or run the network without calling this first will result in errors.
        """
        self.input_size = input_size
        self.A: Tensor = Tensor(np.zeros((self.input_size, 1)), require_grad=False)

        A = self.A
        n = self.input_size

        for layer in self._layers:
            A, n = layer.build(A, n)

        self.output = A
        self.output_size = n

        self._is_built = True
        PrintUtils.print_info(f"[{self.__class__.__name__}] Computation Graph Compiled.")
        PrintUtils.print_info(f"[{self.__class__.__name__}] Parameter Count: {self.utils._get_param_count():,}")

    @requires_build
    # main feed forward function (single)
    def forward(self, input: list) -> list:
        if len(input) != self.input_size:
            raise InputValidationError("Input array size does not match the neural network.")

        A: np.ndarray = self.forward_batch([input], raw_ndarray_output=True)

        return A.flatten().tolist()

    @requires_build
    # main feed forward function (multiple)
    def forward_batch(self, input: list, raw_ndarray_output = False) -> np.ndarray | list:
        if len(input) == 0:
            raise InputValidationError("Input batch does not have data.")
        if len(input[0]) != self.input_size:
            raise InputValidationError("Input array size does not match the neural network.")
        
        current_batch_size = len(input)

        # activation
        self.A.assign(np.array(input).T)

        # setup internal tensors
        self.setup_tensors(current_batch_size, is_training=False)

        # forward pass
        self.output.forward()

        # post updates
        self.sync_after_backward(is_training=True)

        if raw_ndarray_output:
            return self.output.evaluate()
        return self.output.evaluate().T.tolist() # vanilla list, not np.ndarray

    def setup_tensors(self, batch_size: int, is_training = False):
        for layer in self._layers:
            layer.setup_tensors(batch_size, is_training=is_training)

    def sync_after_backward(self, is_training = False):
        for layer in self._layers:
            layer.sync_after_backward(is_training=is_training)