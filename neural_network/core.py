import numpy as np
from .dense_layer import DenseLayer
from .exceptions import InputValidationError
from .metrics import Metrics
from .print_utils import PrintUtils
from .utils import Utils

class NeuralNetwork:

    def __init__(self,
                 layers: list[DenseLayer],
                 loss_function: str = "MSE",
                 learn_rate: float = 0.01,
                 lambda_parem: float = 0.0,
                 momentum: float = 0.8) -> None:
        
        self.utils = Utils(self)
        self.metrics = Metrics(self)
        
        # ===== ===== INPUT VALIDATION START ===== =====
        if not layers: # if list is empty
            raise InputValidationError("Empty layer configuration not possible.")
        
        # Learn Rate
        # How fast or slow this network learns
        #     new_parameter = old_parameter - velocity * learn_rate 
        if learn_rate <= 0.0:
            raise InputValidationError("Learn rate must be positive.")
        if learn_rate >= 0.1:
            PrintUtils.print_warning(f"Warning: Learn rate {learn_rate:.3f} may cause instability. Consider keeping it less than 0.1.")

        # L2 Regularization Strength
        # low reg strength -> cook in class, and fail in exam; overfit
        # high reg strength -> I am dumb dumb, can't learn; underfit
        # Large weights and biases will are penalized more aggressively than small ones
        # Don't set it too large, at most 0.01 (unless you know what you're doing)
        #     new_velocity = momentum * old_velocity + (1-momentum) * (parameter_gradient + lambda_parem * parameter)
        if lambda_parem < 0.0:
            raise InputValidationError("Regularization Strength can't be negative.")
        if lambda_parem > 0.01:
            PrintUtils.print_warning(f"Warning: Regularization Strength {lambda_parem:.3f} is strong. Consider keeping it less than 0.01")

        # Momentum Beta for Momentum Gradient Descent
        # 0.0 disables the momentum behavior
        # having momentum beta helps escape the high loss "plateaus" better
        # high values result in smoother/stronger "gliding" descent
        # MUST be less than 1.0
        #     new_velocity = momentum * old_velocity + (1-momentum) * (parameter_gradient + lambda_parem * parameter)
        if momentum < 0.0:
            raise InputValidationError("Momentum can't be negative.")
        if momentum >= 1.0:
            raise InputValidationError("Momentum must be less than 1.0.")
        if momentum >= 0.95:
            PrintUtils.print_warning(f"Warning: Momentum value {momentum:.3f} may cause strong \"gliding\" behavior. Consider keeping it less than 0.95")

        # Neuron connection check
        for i in range(len(layers) - 1):
            if layers[i].output_size != layers[i + 1].input_size:
                raise InputValidationError(f"Layer {i+1} and {i+2} can't connect.")

        loss_function = loss_function.strip().lower()
        self.utils._loss_func_validator(loss_function)
        # ===== ===== INPUT VALIDATION END ===== =====
        
        self._layers: list[DenseLayer] = layers
        self._layer_count: int = len(layers)
        self.LR = learn_rate
        self.l2_lambda = lambda_parem
        self.m_beta = momentum

        # Activate/Build/Initialize/whatever the layers
        for i in range(self._layer_count):
            if i == 0:
                self._layers[i].build(is_first=True)
                continue
            if i == self._layer_count - 1: # final layer
                self._layers[i].build(is_final=True)
                continue
            self._layers[i].build()

        self._loss_deriv_func = self.utils._get_loss_deriv_func(loss_function)
        
        PrintUtils.print_info(f"Neural network initialization successful.")
        PrintUtils.print_info(f"Parameter Count: {self.utils._get_param_count():,}")

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

            # activation
            # a contains columns of sample inputs, each column is an individual sample
            a: np.ndarray = i_batch.T
            
            # desired output
            # same as a
            y: np.ndarray = o_batch.T

            # forward pass
            for layer in self._layers:
                a: np.ndarray = layer.forward(a)

            # a holds columns of output here
            # y is desired output
            # derivative of loss function with respect to activations for LAST OUTPUT LAYER
            act_grad: np.ndarray = self._loss_deriv_func(a, y)

            # backpropagation to calculate gradients
            for layer in reversed(self._layers):
                act_grad = layer.backward(act_grad)
            
            # apply calculated gradients
            for layer in self._layers:
                layer.optimize(LR=self.LR, l2_lambda=self.l2_lambda, m_beta=self.m_beta)
            
            p: float = (100.0 * (_+1) / epoch)
            print(f"Progress: [{'='*int(30*p/100):<30}] {_+1:>5} / {epoch} [{p:>6.2f}%]  ", end='\r')
        
        PrintUtils.print_success("\n===== ===== ===== Training Completed ===== ===== =====")
