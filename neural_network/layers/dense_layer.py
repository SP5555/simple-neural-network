import numpy as np
from ..activations.activation import Activation
from ..auto_diff.auto_diff_reverse import Tensor, Matmul
from ..exceptions import InputValidationError
from .layer import Layer
from ..common import ParamDict

class DenseLayer(Layer):
    """
    A fully connected (dense) layer where every input neuron 
    is connected to every output neuron.

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
    
    weight_decay : float, optional
        Strength of L2 regularization.
        Default is 0.0, meaning no regularization.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Activation,
                 weight_decay: float = 0.0) -> None:

        super().__init__(input_size, output_size, activation, weight_decay)

    def build(self, is_first: bool = False, is_final: bool = False) -> None:

        # helpers, they save a lot of power
        self._is_first = is_first
        self._is_final = is_final

        if not is_final and self.activation.is_LL_exclusive:
            raise InputValidationError(f"{self.activation.__class__.__name__} activation can't be used in hidden layers.")

        # auto-diff tensor objects
        self._W: Tensor = Tensor(np.random.randn(self.output_size, self.input_size) * np.sqrt(2/self.input_size))
        self._B: Tensor = Tensor(np.random.randn(self.output_size, 1))
 
        # gradient container numpy array objects
        self._W_grad = None
        self._B_grad = None

        # tmp vars
        self.tmp_batch_size = None

    def compile(self, A: Tensor) -> Tensor:

        # Z = W*A_in + B
        _Z = Matmul(self._W, A) + self._B

        # A_out = activation(Z)
        self.activation.build_expression(_Z)

        self._out = self.activation.expression
        return self._out

    def setup_tensors(self, batch_size: int, is_training: bool = False):

        self.tmp_batch_size = batch_size

    # collects grad from Tensors
    def regularize_grads(self) -> np.ndarray:

        self._W_grad = self._W.grad / self.tmp_batch_size
        _W_grad_l2: np.ndarray = self._W.tensor * self.L2_lambda # Compute regularization term
        self._W_grad += _W_grad_l2

        self._B_grad = np.sum(self._B.grad, axis=1, keepdims=True) / self.tmp_batch_size
        _B_grad_l2: np.ndarray = self._B.tensor * self.L2_lambda # Compute regularization term
        self._B_grad += _B_grad_l2

        if self.activation.is_learnable:

            self.activation._alpha_grad = np.sum(self.activation._alpha.grad, axis=1, keepdims=True) / self.tmp_batch_size
            _alpha_grad_l2: np.ndarray = self.activation._alpha.tensor * self.L2_lambda # Compute regularization term
            self.activation._alpha_grad += _alpha_grad_l2

    def _get_weights_and_grads(self) -> list[ParamDict]:
        params = [
            {'weight': self._W, 'grad': self._W_grad},
            {'weight': self._B, 'grad': self._B_grad}
        ]
        if self.activation.is_learnable:
            params.append({
                'weight': self.activation._alpha,
                'grad': self.activation._alpha_grad,
                'learnable': True,
                'constraints': self.activation._alpha_constraints
            })
        return params

    def zero_grads(self):
        self._W.zero_grad()
        self._B.zero_grad()
        if self.activation.is_learnable:
            self.activation._alpha.zero_grad()

    def _get_param_count(self) -> int:
        w = self.input_size * self.output_size
        s = self.output_size
        lp = self.output_size if self.activation.is_learnable else 0
        return w + s + lp