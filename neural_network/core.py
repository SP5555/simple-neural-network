import numpy as np
from .exceptions import InputValidationError
from .losses.loss import Loss
from .layers.layer import Layer
from .metrics import Metrics
from .optimizers.optimizer import Optimizer
from .print_utils import PrintUtils
from .utils import Utils

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

    optimizer : Optimizer
        An instance of a class derived from the Optimizer base class.

    loss_function : Loss
        An instance of a class derived from the Loss base class for training.
    """
    def __init__(self,
                 layers: list[Layer],
                 optimizer: Optimizer = None,
                 loss_function: Loss = None) -> None:
        
        self.utils = Utils(self)
        self.metrics = Metrics(self)
        
        # ===== ===== INPUT VALIDATION START ===== =====
        if not layers: # if list is empty
            raise InputValidationError("Empty layer configuration not possible.")

        # Neuron connection check
        for i in range(len(layers) - 1):
            if layers[i].output_size != layers[i + 1].input_size:
                raise InputValidationError(f"Layer {i+1} and {i+2} can't connect.")
        
        if not optimizer:
            raise InputValidationError("Neural Network is missing an optimizer.")

        if not loss_function:
            raise InputValidationError("Neural Network is missing a loss function.")
        # ===== ===== INPUT VALIDATION END ===== =====
        
        self._layers: list[Layer] = layers
        self._layer_count: int = len(layers)
        self._loss_func = loss_function

        # Activate/Build/Initialize/whatever the layers
        for i in range(self._layer_count):
            if i == 0:
                self._layers[i].build(is_first=True)
                continue
            if i == self._layer_count - 1: # final layer
                self._layers[i].build(is_final=True)
                continue
            self._layers[i].build()
        PrintUtils.print_info(f"[{self.__class__.__name__}] Layers built.")

        self.optimizer = optimizer
        PrintUtils.print_info(f"[{self.__class__.__name__}] {self.optimizer.__class__.__name__} Optimizer initialized.")

        PrintUtils.print_info(f"[{self.__class__.__name__}] Neural network initialized.")
        PrintUtils.print_info(f"[{self.__class__.__name__}] Parameter Count: {self.utils._get_param_count():,}")

    # main feed forward function (single)
    def forward(self, input: list) -> list:
        if len(input) != self._layers[0].input_size:
            raise InputValidationError("Input array size does not match the neural network.")

        a: np.ndarray = self.forward_batch([input], raw_ndarray_output=True)

        return a.flatten().tolist()

    # main feed forward function (multiple)
    def forward_batch(self, input: list, raw_ndarray_output = False) -> np.ndarray:
        if len(input) == 0:
            raise InputValidationError("Input batch does not have data.")
        if len(input[0]) != self._layers[0].input_size:
            raise InputValidationError("Input array size does not match the neural network.")
        
        # activation
        # changes an array of inputs into n x batch_size numpy 2D array
        a: np.ndarray = np.array(input).T

        # forward pass
        for layer in self._layers:
            a: np.ndarray = layer.forward(a)

        if raw_ndarray_output:
            return a
        return a.T.tolist() # vanilla list, not np.ndarray

    def train(self,
              input_list: list,
              output_list: list,
              epoch: int = 100,
              batch_size: int = 32) -> None:
        if len(input_list) == 0 or len(output_list) == 0:
            raise InputValidationError("Datasets can't be empty.")
        if len(input_list) != len(output_list):
            raise InputValidationError("Input and Output data set sizes must be equal.")
        if len(input_list[0]) != self._layers[0].input_size:
            raise InputValidationError("Input array size does not match the neural network.")
        if len(output_list[0]) != self._layers[-1].output_size:
            raise InputValidationError("Output array size does not match the neural network.")
        if epoch <= 0:
            raise InputValidationError("Epoch must be positive.")
        
        if batch_size > len(input_list): batch_size = len(input_list)

        input_ndarray = np.array(input_list)
        output_ndarray = np.array(output_list)
        
        for _ in range(epoch):
            # pick random candidates as train data in each epoch
            indices = np.random.choice(len(input_list), size=batch_size, replace=False)
            i_batch = input_ndarray[indices]
            o_batch = output_ndarray[indices]

            # activation (input_size, batch_size)
            a: np.ndarray = i_batch.T
            
            # desired output (output_size, batch_size)
            y: np.ndarray = o_batch.T

            # FORWARD PASS: compute activations
            for layer in self._layers:
                a: np.ndarray = layer.forward(a, is_training=True)
            # dims of a after forward pass: (output_size, batch_size)

            # derivative of loss function with respect to activations for LAST OUTPUT LAYER
            act_grad: np.ndarray = self._loss_func.grad(a, y)

            # BACKPROPAGATION: calculate gradients
            for layer in reversed(self._layers):
                act_grad = layer.backward(act_grad)

            # collect params to pass into optimizer
            all_params = []
            for layer in self._layers:
                all_params.extend(layer._get_params())

            # OPTIMIZATION: apply gradients
            self.optimizer.step(all_params)

            p: float = (100.0 * (_+1) / epoch)
            print(f"Progress: [{'='*int(30*p/100):<30}] {_+1:>5} / {epoch} [{p:>6.2f}%]  ", end='\r')

        PrintUtils.print_success("\n===== ===== ===== Training Completed ===== ===== =====")
