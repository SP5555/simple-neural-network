import numpy as np
from .auto_diff.auto_diff_reverse import Tensor
from .common import PrintUtils, InputValidationError
from .core import NeuralNetwork
from .losses.loss import Loss
from .optimizers.optimizer import Optimizer

class Trainer:
    """
    Trainer
    =====
    ???

    Parameters
    ----------
    model : NeuralNetwork
        Neural Network to be trained.

    optimizer : Optimizer
        An instance of a class derived from the Optimizer base class.

    loss_function : Loss
        An instance of a class derived from the Loss base class for training.
    """
    def __init__(self,
                 model: NeuralNetwork,
                 loss_function: Loss,
                 optimizer: Optimizer,
                 verbose: bool = True):

        # ===== ===== INPUT VALIDATION START ===== =====
        if not model._is_built:
            raise Exception("Neural Network is not built.")

        if not loss_function:
            raise InputValidationError("Neural Network is missing a loss function.")

        if not optimizer:
            raise InputValidationError("Neural Network is missing an optimizer.")
        # ===== ===== INPUT VALIDATION END ===== =====

        self.model = model
        self.loss_func = loss_function
        self.optimizer = optimizer
        PrintUtils.print_info(f"[{self.__class__.__name__}] {self.optimizer.__class__.__name__} Optimizer initialized.")

        self.print_progress = PrintUtils.print_progress_bar if verbose else PrintUtils.print_noop

        self.verbose = verbose

        # ===== Connect Loss at the end of Computation Graph =====
        self.output_target: Tensor = Tensor(np.zeros((self.model.output_size, 1)), require_grad=False)
        self.loss_func.build_expression(self.model.output, self.output_target)
        PrintUtils.print_info(f"[{self.__class__.__name__}] Computation Graph Compiled.")

        PrintUtils.print_info(f"[{self.__class__.__name__}] Trainer initialized.")
    
    def train(self,
              input_list: list,
              output_list: list,
              epoch: int = 100,
              batch_size: int = 32) -> None:
        if len(input_list) == 0 or len(output_list) == 0:
            raise InputValidationError("Datasets can't be empty.")
        if len(input_list) != len(output_list):
            raise InputValidationError("Input and Output data set sizes must be equal.")
        if len(input_list[0]) != self.model.input_size:
            raise InputValidationError("Input array size does not match the neural network.")
        if len(output_list[0]) != self.model.output_size:
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

            self.train_step(i_batch, o_batch)

            self.print_progress(_+1, epoch)

        PrintUtils.print_success("\n===== ===== ===== Training Completed ===== ===== =====")

    def train_step(self, i_batch: np.ndarray, o_batch: np.ndarray):

            current_batch_size = len(i_batch)

            # input features
            self.model.A.assign(i_batch.T)

            # target output
            self.output_target.assign(o_batch.T)

            # setup internal tensors
            self.setup_tensors(current_batch_size, is_training=True)

            # FORWARD PASS: calculate forward values (LITTLE MAGIC)
            # auto diff forward call
            # situates all tensors/computation nodes with their values
            self.loss_func.forward()

            # BACKPROPAGATION: calculate gradients (BIG MAGIC)
            # auto diff reverse mode backward call
            # situates all tensors with their gradients
            seed: np.ndarray = np.ones_like(self.output_target.tensor)
            self.loss_func.backward(seed)

            # collect params to pass into optimizer
            weights_and_grads = []
            for layer in self.model._layers:
                layer.prepare_grads()
                weights_and_grads.extend(layer._get_weights_and_grads())
                layer.zero_grads()

            # OPTIMIZATION: apply gradients
            self.optimizer.step(weights_and_grads)

    def setup_tensors(self, batch_size: int, is_training=False):
        for layer in self.model._layers:
            layer.setup_tensors(batch_size, is_training=is_training)