import numpy as np
from .dense_layer import DenseLayer
from ..activations import Activations
from ..exceptions import InputValidationError
from ..print_utils import PrintUtils
from ..utils import Utils

class DropoutLayer(DenseLayer):
    
    def __init__(self, input_size: int, output_size: int, activation: str, dropout_probability: float) -> None:
        super().__init__(input_size, output_size, activation)

        if dropout_probability < 0.0 or dropout_probability > 1.0:
            raise InputValidationError("Dropout Probability can't be less than must be within 0.0 and 1.0")
        if dropout_probability > 0.5:
            PrintUtils.print_warning(f"Dropout Probability of {dropout_probability} is too high. Consider less than 0.5")
        
        self.dp = dropout_probability
    
    # compute a layer's output based on the input.
    def forward(self, input: np.ndarray, is_training: bool = False) -> np.ndarray:
        
        self._a_in: np.ndarray = input
        # z = W*A + b
        self._z: np.ndarray = np.matmul(self.weights, self._a_in) + self.biases.reshape(-1, 1) # broadcasting
        # A = activation(z)
        self._a: np.ndarray = self._act_func(self._z, self.alpha)

        if is_training:
            # create a mask where each neuron has a 1-dp chance to remain active
            mask = np.random.binomial(n=1, p=1-self.dp, size=self._a.shape)
            
            # Apply dropout
            # zero out p fraction of activations and scale up the surviving activations
            self._a *= mask / (1-self.dp)

        return self._a
    
    @property
    def requires_training_flag(self):
        return True