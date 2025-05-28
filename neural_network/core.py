import numpy as np
from .auto_diff.auto_diff_reverse import Tensor
from .common import Metrics, Utils, PrintUtils, requires_build, InputValidationError
from .layers.layer import Layer
from .layers.regularizablelayer import RegularizableLayer

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

    weight_decay : float, optional
        Global weight decay parameter used as default for all compatible layers unless overridden. \\
        Default is `0.0`.
    """
    def __init__(self,
                 layers: list[Layer],
                 weight_decay: float | None = None) -> None:
        
        self.utils = Utils(self)
        self.metrics = Metrics(self)
        
        # ===== ===== INPUT VALIDATION START ===== =====
        if not layers: # if list is empty
            raise InputValidationError("Empty layer configuration not possible.")
        
        if weight_decay is not None:
            if weight_decay < 0.0:
                raise InputValidationError("Regularization Strength can't be negative.")
            if weight_decay > 0.01:
                PrintUtils.print_warning(f"[{self.__class__.__name__}] Warning: " + 
                                        f"Regularization Strength {weight_decay:.3f} is strong. " +
                                        "Consider keeping it less than 0.01")
        # ===== ===== INPUT VALIDATION END ===== =====
        
        self._layers: list[Layer] = layers
        self._layer_count: int = len(layers)

        self._weight_decay: float = weight_decay

        PrintUtils.print_info(f"[{self.__class__.__name__}] Neural network initialized.")

        self._is_built = False
    
    def build(self, input_size: int):
        """
        Builds necessary internal tensors and compiles the **computation graph** for the neural network.

        This function ensures that all layers are connected properly and constructs 
        the forward computation path only once, improving efficiency during training 
        and inference by avoiding redundant graph re-construction.
        
        Parameters
        ----------
        input_size : int
            The number of input features (i.e., the size of the input vector to the first layer). \\
            For example, use 4 if your dataset samples are shaped like (4,).

        Notes
        -----
        - This function **must** be called before training or inference.
        - Attempting to train or use the network without calling this first will raise an error.
        """
        self.input_size = input_size
        self.A: Tensor = Tensor(np.zeros((self.input_size, 1)), requires_grad=False)

        A = self.A
        n = self.input_size

        for layer in self._layers:
            if isinstance(layer, RegularizableLayer):
                A, n = layer.build(A, n, self._weight_decay)
            else:
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
        self.A.assign(np.array(input).T)

        self.forward_pass(current_batch_size, is_training=False)

        if raw_ndarray_output:
            return self.output.evaluate()
        return self.output.evaluate().T.tolist() # vanilla list, not np.ndarray

    def forward_pass(self, batch_size: int, is_training = False):

        self.pre_setup_tensors(batch_size, is_training)
        
        self.output.forward()

        self.post_setup_tensors(is_training)

    def pre_setup_tensors(self, batch_size: int, is_training=False):
        for layer in self._layers:
            layer.pre_setup_tensors(batch_size, is_training=is_training)

    def post_setup_tensors(self, is_training = False):
        for layer in self._layers:
            layer.post_setup_tensors(is_training=is_training)