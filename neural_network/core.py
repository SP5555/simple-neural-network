import numpy as np
from .auto_diff.auto_diff_reverse import Tensor
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

        self.compile_graph()
        PrintUtils.print_info(f"[{self.__class__.__name__}] Computation Graph Compiled.")

        self.optimizer = optimizer
        PrintUtils.print_info(f"[{self.__class__.__name__}] {self.optimizer.__class__.__name__} Optimizer initialized.")

        PrintUtils.print_info(f"[{self.__class__.__name__}] Neural network initialized.")
        PrintUtils.print_info(f"[{self.__class__.__name__}] Parameter Count: {self.utils._get_param_count():,}")
    
    def compile_graph(self):
        """
        Compile the **computation graph** for the neural network.

        This function ensures that all layers are connected properly and constructs 
        the forward computation path only once, improving efficiency during training 
        and inference by avoiding redundant graph re-construction.

        Notes:
        - This function **must** be called before training or inference.
        - Any attempt to train or run the network without calling this first will result in errors.
        """
        self.A: Tensor = Tensor(np.zeros((self._layers[0].input_size, 1)), require_grad=False)
        self.Y: Tensor = Tensor(np.zeros((self._layers[-1].output_size, 1)), require_grad=False)

        A = self.A

        for layer in self._layers:
            A = layer.compile(A)
        self._loss_func.build_expression(A, self.Y)

    # main feed forward function (single)
    def forward(self, input: list) -> list:
        if len(input) != self._layers[0].input_size:
            raise InputValidationError("Input array size does not match the neural network.")

        A: np.ndarray = self.forward_batch([input], raw_ndarray_output=True)

        return A.flatten().tolist()

    # main feed forward function (multiple)
    def forward_batch(self, input: list, raw_ndarray_output = False) -> np.ndarray:
        if len(input) == 0:
            raise InputValidationError("Input batch does not have data.")
        if len(input[0]) != self._layers[0].input_size:
            raise InputValidationError("Input array size does not match the neural network.")
        
        current_batch_size = len(input)

        # activation
        # changes an array of inputs into n x batch_size numpy 2D array
        self.A.tensor = np.array(input).T

        # forward pass
        for layer in self._layers:
            layer.tmp_batch_size = current_batch_size
            layer.setup_tensors()
        self._layers[-1].forward()

        if raw_ndarray_output:
            return self._layers[-1]._out.evaluate()
        return self._layers[-1]._out.evaluate().T.tolist() # vanilla list, not np.ndarray

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

            current_batch_size = len(i_batch)

            # input features
            self.A.tensor = i_batch.T

            # target output
            self.Y.tensor = o_batch.T

            # setup necessary tensors for the forward call
            for layer in self._layers:
                layer.tmp_batch_size = current_batch_size
                layer.setup_tensors(is_training=True)

            # FORWARD PASS: calculate forward values
            # auto diff forward call
            # situates all tensors/computation nodes with their values
            self._loss_func.forward()

            # BACKPROPAGATION: calculate gradients (MAGIC)
            # auto diff reverse mode backward call
            # situates all tensors with their gradients
            seed: np.ndarray = np.ones_like(self.Y.tensor)
            self._loss_func.backward(seed)

            # collect params to pass into optimizer
            weights_and_grads = []
            for layer in self._layers:
                layer.regularize_grads()
                weights_and_grads.extend(layer._get_weights_and_grads())
                layer.zero_grads()

            # OPTIMIZATION: apply gradients
            self.optimizer.step(weights_and_grads)

            p: float = (100.0 * (_+1) / epoch)
            print(f"Progress: [{'='*int(30*p/100):<30}] {_+1:>5} / {epoch} [{p:>6.2f}%]  ", end='\r')

        PrintUtils.print_success("\n===== ===== ===== Training Completed ===== ===== =====")
