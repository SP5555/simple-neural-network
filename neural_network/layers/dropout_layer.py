import numpy as np
from .dense_layer import DenseLayer
from ..activations import Activations
from ..exceptions import InputValidationError
from ..print_utils import PrintUtils

class DropoutLayer(DenseLayer):
    
    def __init__(self, input_size: int, output_size: int, activation: str, dropout_probability: float, batch_wise=False) -> None:
        super().__init__(input_size, output_size, activation)

        if dropout_probability < 0.0 or dropout_probability > 1.0:
            raise InputValidationError("Dropout Probability can't be less than must be within 0.0 and 1.0")
        if dropout_probability > 0.5:
            PrintUtils.print_warning(f"Dropout Probability of {dropout_probability} is too high. Consider less than 0.5")
        
        self.dp = dropout_probability
        self.batch_wise = batch_wise

    def build(self, is_first: bool = False, is_final: bool = False):
        if is_final:
            PrintUtils.print_warning("Using a dropout layer as the final layer is not recommended.")
        if self.act_name in Activations._dropout_incomp_acts:
            raise InputValidationError(f"{self.act_name} is not compatible in the dropout layer.")
        
        super().build(is_first, is_final)
    
    # compute a layer's output based on the input.
    def forward(self, input: np.ndarray, is_training: bool = False) -> np.ndarray:
        
        self._a_in: np.ndarray = input
        # z = W*A_in + b
        self._z: np.ndarray = np.matmul(self.weights, self._a_in) + self.biases # auto-broadcasting
        # A_out = activation(z, learn_b)
        self._a: np.ndarray = self._act_func(self._z, self.alpha)

        if is_training:
            # standard dropout: randomly drops neurons individually within each sample
            # batch-wise dropout: same dropout pattern to all neurons within a mini-batch
            shape = self._a.shape
            if self.batch_wise:
                shape = (self._a.shape[0], 1)
            # create a mask where a neuron has a 1-dp chance to remain active
            mask = np.random.binomial(n=1, p=1-self.dp, size=shape)
            
            # Apply dropout
            # zero out dp fraction of activations and scale up the surviving activations
            self._a *= mask / (1-self.dp)

        return self._a
    
    @property
    def requires_training_flag(self):
        return True