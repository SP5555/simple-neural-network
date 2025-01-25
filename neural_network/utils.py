import numpy as np
from .activations import Activations
from .losses import Losses
from .exceptions import InputValidationError

class Utils:
    def __init__(self, core_instance):
        self.core = core_instance
 
    def inspect_weights_and_biases(self) -> None:
        np.set_printoptions(precision=4)
        for i in range(self.core._layer_count - 1):
            print(f'w L{i+1} -> L{i+2}')
            print(self.core.weights[i])
            print(f'b L{i+1} -> L{i+2}')
            print(self.core.biases[i])

    def get_parameter_count(self) -> int:
        c: int = 0
        for i in range(self.core._layer_count - 1):
            # c += self.layers[i + 1] * self.layers[i] # Weights
            # c += self.layers[i + 1] # Biases
            c += self.core.layers[i + 1] * (self.core.layers[i] + 1)
        return c

    @staticmethod
    def get_activation_func(name: str):
        actv_funcs = {
            'relu': Activations.relu,
            'leaky_relu': Activations.leaky_relu,
            'tanh': Activations.tanh,
            'sigmoid': Activations.sigmoid,
            'id': Activations.id, 'linear': Activations.id,
            'softmax': Activations.softmax
        }
        name = name.strip().lower()
        if name in actv_funcs: return actv_funcs[name]
        raise InputValidationError(f"Unsupported activation function: {name}")

    @staticmethod
    def get_activation_derivative_func(name: str):
        actv_deriv_funcs = {
            'relu': Activations.relu_derivative,
            'leaky_relu': Activations.leaky_relu_derivative,
            'tanh': Activations.tanh_derivative,
            'sigmoid': Activations.sigmoid_derivative,
            'id': Activations.id_derivative, 'linear': Activations.id_derivative,
            'softmax': Activations.softmax_derivative
        }
        name = name.strip().lower()
        if name in actv_deriv_funcs: return actv_deriv_funcs[name]
        raise InputValidationError(f"Unsupported activation function: {name}")

    @staticmethod
    def get_loss_derivative_func(name: str):
        loss_funcs = {
            'mse': Losses.MSE_gradient,
            'bce': Losses.BCE_gradient,
            'mce': Losses.MCE_gradient, 'cce': Losses.MCE_gradient
        }
        name = name.strip().lower()
        if name in loss_funcs: return loss_funcs[name]
        raise InputValidationError(f"Unsupported loss function: {name}")
