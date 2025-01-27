import numpy as np
from .activations import Activations, ActivationWrapper
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
            print(f'w L{i+1} -> L{i+2}:')
            print(self.core.weights[i])
            print(f'b L{i+1} -> L{i+2}:')
            print(self.core.biases[i])
            print(f'learnable param L{i+1} -> L{i+2}:')
            print(self.core.alpha[i])

    def _get_param_count(self) -> int:
        c: int = 0
        for i in range(self.core._layer_count - 1):
            # c += self._layers[i + 1] * self._layers[i] # Weights
            # c += self._layers[i + 1] # Biases
            c += self.core._layers[i + 1] * (self.core._layers[i] + 1)
            if (self.core._act_func[i].name in Activations._learnable_acts):
                c += self.core._layers[i + 1]
        return c

    @staticmethod
    def _get_act_func(name: str):
        actv_funcs = {
            'relu':         ActivationWrapper(lambda z, b: Activations._relu(z), "relu"),
            'leaky_relu':   ActivationWrapper(lambda z, b: Activations._leaky_relu(z), "leaky_relu"),
            'prelu':        ActivationWrapper(lambda z, b: Activations._prelu(z, b), "prelu"),
            'tanh':         ActivationWrapper(lambda z, b: Activations._tanh(z), "tanh"),
            'sigmoid':      ActivationWrapper(lambda z, b: Activations._sigmoid(z), "sigmoid"),
            'swish':        ActivationWrapper(lambda z, b: Activations._swish(z, b), "swish"),
            'swish_f':      ActivationWrapper(lambda z, b: Activations._swish_fixed(z), "swish_f"),
            'id':           ActivationWrapper(lambda z, b: Activations._id(z), "id"),
            'linear':       ActivationWrapper(lambda z, b: Activations._id(z), "linear"),
            'softmax':      ActivationWrapper(lambda z, b: Activations._softmax(z), "softmax"),
        }
        name = name.strip().lower()
        if name in actv_funcs: return actv_funcs[name]
        raise InputValidationError(f"Unsupported activation function: {name}")

    @staticmethod
    def _get_act_deriv_func(name: str):
        actv_deriv_funcs = {
            'relu':         ActivationWrapper(lambda z, b: Activations._relu_deriv(z), "relu"),
            'leaky_relu':   ActivationWrapper(lambda z, b: Activations._leaky_relu_deriv(z), "leaky_relu"),
            'prelu':        ActivationWrapper(lambda z, b: Activations._prelu_deriv(z, b), "prelu"),
            'tanh':         ActivationWrapper(lambda z, b: Activations._tanh_deriv(z), "tanh"),
            'sigmoid':      ActivationWrapper(lambda z, b: Activations._sigmoid_deriv(z), "sigmoid"),
            'swish':        ActivationWrapper(lambda z, b: Activations._swish_deriv(z, b), "swish"),
            'swish_f':      ActivationWrapper(lambda z, b: Activations._swish_fixed_deriv(z), "swish_f"),
            'id':           ActivationWrapper(lambda z, b: Activations._id_deriv(z), "id"),
            'linear':       ActivationWrapper(lambda z, b: Activations._id_deriv(z), "linear"),
            'softmax':      ActivationWrapper(lambda z, b: Activations._softmax_deriv(z), "softmax"),
        }
        name = name.strip().lower()
        if name in actv_deriv_funcs: return actv_deriv_funcs[name]
        raise InputValidationError(f"Unsupported activation function: {name}")

    @staticmethod
    def _get_learnable_alpha_grad_func(name: str):
        learnable_alpha_grad_funcs = {
            'swish':        ActivationWrapper(lambda z, b: Activations._swish_param_deriv(z, b), "swish"),
            'prelu':        ActivationWrapper(lambda z, b: Activations._prelu_param_deriv(z, b), "prelu"),
        }
        name = name.strip().lower()
        if name in learnable_alpha_grad_funcs:
            return learnable_alpha_grad_funcs[name]
        return ActivationWrapper(lambda z, b=None: 0, "N/A")

    @staticmethod
    def _get_loss_deriv_func(name: str):
        loss_funcs = {
            'mse': Losses._mse_grad,
            'bce': Losses._bce_grad,
            'cce': Losses._mce_grad
        }
        name = name.strip().lower()
        if name in loss_funcs: return loss_funcs[name]
        raise InputValidationError(f"Unsupported loss function: {name}")
