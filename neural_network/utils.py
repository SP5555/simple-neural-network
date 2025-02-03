import numpy as np
from .activations import Activations, ActivationWrapper
from .exceptions import InputValidationError
from .losses import Losses
from .print_utils import PrintUtils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import NeuralNetwork

class Utils:
    def __init__(self, core_instance: "NeuralNetwork"):
        self.core = core_instance

    def inspect_weights_and_biases(self) -> None:
        np.set_printoptions(precision=4)
        for i, layer in enumerate(self.core._layers):
            PrintUtils.print_info(f'weights L{i+1} -> L{i+2}:')
            print(layer.weights)
            PrintUtils.print_info(f'biases L{i+1} -> L{i+2}:')
            print(layer.biases)
            PrintUtils.print_info(f'learnable param L{i+1} -> L{i+2}:')
            print(layer.alpha)
    
    def _get_param_count(self) -> int:
        c: int = 0
        for layer in self.core._layers:
            c += layer._get_param_count()
        return c
    
    # names must be already all lowercase
    @staticmethod
    def _act_func_validator(name: str):
        if name not in Activations._supported_acts:
            raise InputValidationError(f"Unsupported activation function: {name}")

    @staticmethod
    def _loss_func_validator(name: str):
        if name not in Losses._supported_loss:
            raise InputValidationError(f"Unsupported loss function: {name}")

    # === for all activation related functions =====
    # z (logits; raw score) comes in with the dimensions of (n, batch_size)
    # b (learnable parameter) comes in with the dimensions of (n, 1)
    # b contains None if function is not learnable; appropriate values if learnable
    # b is auto-broadcasted by element-wise operations inside the functions
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
        return actv_funcs[name]

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
        return actv_deriv_funcs[name]

    @staticmethod
    def _get_learnable_alpha_grad_func(name: str):
        learnable_alpha_grad_funcs = {
            'swish':        ActivationWrapper(lambda z, b: Activations._swish_param_deriv(z, b), "swish"),
            'prelu':        ActivationWrapper(lambda z, b: Activations._prelu_param_deriv(z, b), "prelu"),
        }
        if name in learnable_alpha_grad_funcs:
            return learnable_alpha_grad_funcs[name]
        return ActivationWrapper(lambda z, b: 0, "N/A")

    @staticmethod
    def _get_loss_deriv_func(name: str):
        loss_funcs = {
            'mse': Losses._mse_grad,
            'bce': Losses._bce_grad,
            'cce': Losses._cce_grad
        }
        return loss_funcs[name]
