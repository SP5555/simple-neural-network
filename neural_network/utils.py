import numpy as np
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
            print(layer._W.tensor)
            PrintUtils.print_info(f'biases L{i+1} -> L{i+2}:')
            print(layer._B.tensor)
            if layer.activation.is_learnable:
                PrintUtils.print_info(f'learnable param L{i+1} -> L{i+2}:')
                print(layer.activation._alpha.tensor)
    
    def _get_param_count(self) -> int:
        c: int = 0
        for layer in self.core._layers:
            c += layer._get_param_count()
        return c
