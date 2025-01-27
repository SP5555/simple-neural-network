import numpy as np
from .activations import Activations
from .losses import Losses
from .exceptions import InputValidationError

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import NeuralNetwork

class Utils:
    def __init__(self, core_instance: "NeuralNetwork"):
        self.core = core_instance
 
    def inspect_weights_and_biases(self) -> None:
        np.set_printoptions(precision=4)
        for i in range(self.core._layer_count - 1):
            print(f'w L{i+1} -> L{i+2}')
            print(self.core.weights[i])
            print(f'b L{i+1} -> L{i+2}')
            print(self.core.biases[i])

    def _get_param_count(self) -> int:
        c: int = 0
        for i in range(self.core._layer_count - 1):
            # c += self.layers[i + 1] * self.layers[i] # Weights
            # c += self.layers[i + 1] # Biases
            c += self.core._layers[i + 1] * (self.core._layers[i] + 1)
        return c

    @staticmethod
    def _get_act_func(name: str):
        actv_funcs = {
            'relu': Activations._relu,
            'leaky_relu': Activations._leaky_relu,
            'tanh': Activations._tanh,
            'sigmoid': Activations._sigmoid,
            'swish': Activations._swish,
            'id': Activations._id, 'linear': Activations._id,
            'softmax': Activations._softmax
        }
        name = name.strip().lower()
        if name in actv_funcs: return actv_funcs[name]
        raise InputValidationError(f"Unsupported activation function: {name}")

    @staticmethod
    def _get_act_deriv_func(name: str):
        actv_deriv_funcs = {
            'relu': Activations._relu_deriv,
            'leaky_relu': Activations._leaky_relu_deriv,
            'tanh': Activations._tanh_deriv,
            'sigmoid': Activations._sigmoid_deriv,
            'swish': Activations._swish_deriv,
            'id': Activations._id_deriv, 'linear': Activations._id_deriv,
            'softmax': Activations._softmax_deriv
        }
        name = name.strip().lower()
        if name in actv_deriv_funcs: return actv_deriv_funcs[name]
        raise InputValidationError(f"Unsupported activation function: {name}")

    @staticmethod
    def _get_loss_deriv_func(name: str):
        loss_funcs = {
            'mse': Losses._mse_grad,
            'bce': Losses._bce_grad,
            'mce': Losses._mce_grad, 'cce': Losses._mce_grad
        }
        name = name.strip().lower()
        if name in loss_funcs: return loss_funcs[name]
        raise InputValidationError(f"Unsupported loss function: {name}")
