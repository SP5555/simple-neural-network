import numpy as np
from ..activations.activation import Activation
from ..auto_diff.auto_diff_reverse import Tensor, Matmul
from ..exceptions import InputValidationError
from ..print_utils import PrintUtils
from .dense_layer import DenseLayer

class DropoutLayer(DenseLayer):
    """
    A fully connected layer with dropout regularization technique.

    Parameters
    ----------
    input_size : int
        Number of input neurons. Must match the output size of the previous layer
        or the input dimension if this is the first layer.

    output_size : int
        Number of output neurons. Must match the input size of the next layer
        or the final output dimension if this is the last layer.

    activation : Activation
        Activation function to apply to the output neurons.

    dropout_rate : float
        Fraction of neurons to drop during training. Should be between 0.0
        (no dropout) and 1.0 (drop everything, never do this).
        Typical values are between 0.2 and 0.5.

    batch_wise : bool, optional
        If True, the same dropout mask is applied across the entire batch.
        If False, dropout is applied independently to each sample.
        Default is `False`.

    weight_decay : float, optional
        Strength of L2 regularization.
        Default is 0.0, meaning no regularization.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Activation,
                 dropout_rate: float,
                 batch_wise: bool = False,
                 weight_decay: float = 0.0) -> None:

        super().__init__(input_size, output_size, activation, weight_decay)

        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise InputValidationError("Dropout Probability can't be less than must be within 0.0 and 1.0")
        if dropout_rate > 0.5:
            PrintUtils.print_warning(f"Dropout Probability of {dropout_rate} is too high. Consider less than 0.5")
        
        self.dp = dropout_rate
        self.batch_wise = batch_wise

    def build(self, is_first: bool = False, is_final: bool = False):
        if is_final:
            PrintUtils.print_warning("Using a dropout layer as the final layer is not recommended.")
        if self.activation.is_dropout_incompatible:
            raise InputValidationError(f"{self.activation.__class__.__name__} is not compatible in the dropout layer.")
        
        super().build(is_first, is_final)

    def compile(self, A: Tensor) -> Tensor:

        self._A: Tensor = A
        self._W: Tensor = Tensor(self.weights)
        # biases have some broadcast risk, hope numpy auto broadcast works
        self._B: Tensor = Tensor(self.biases)

        # Z = W*A_in + B
        _Z = Matmul(self._W, self._A) + self._B

        # A_out = activation(Z)
        self.activation.build_expression(_Z)

        self.mask = Tensor(1.0, require_grad=False)
        self.rescaler = Tensor(1.0, require_grad=False)

        # Apply dropout
        # zero out dp fraction of activations and scale up the surviving activations
        self._out = self.activation.expression * self.mask * self.rescaler
        return self._out

    def setup_tensors(self, is_training: bool = False):

        # self.tmp_batch_size = self._A.tensor.shape[1]
        self._W.tensor = self.weights
        self._B.tensor = np.repeat(self.biases, self.tmp_batch_size, axis=1)

        if is_training:
            # standard dropout: randomly drops neurons individually within each sample
            # batch-wise dropout: same dropout pattern to all neurons within a mini-batch
            shape = (self.output_size, self.tmp_batch_size)
            if self.batch_wise:
                shape = (self.output_size, 1)
            # create a mask where a neuron has a 1-dp chance to remain active
            self.mask.tensor = np.random.binomial(n=1, p=1-self.dp, size=shape)
            self.rescaler.tensor = 1.0 / (1.0 - self.dp)
        else:
            self.mask.tensor = 1.0
            self.rescaler.tensor = 1.0