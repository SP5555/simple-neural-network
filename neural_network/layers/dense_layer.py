import numpy as np
from ..activations.activation import Activation
from ..auto_diff.auto_diff_reverse import Tensor, Matmul
from ..exceptions import InputValidationError
from .layer import Layer

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

        self.weights: np.ndarray = np.random.randn(self.output_size, self.input_size) * np.sqrt(2/self.input_size)
        self.biases: np.ndarray = np.random.randn(self.output_size, 1)
 
        # initialized with None for efficiency
        # gradient container numpy array objects
        self._W_grad = None
        self._B_grad = None

        # tensor/operation auto-diff objects
        self._W = None  # shape: (output_size, input_size)
        self._B = None  # shape: (output_size, 1) broadcast to (output_size, batch_size)

        # tmp vars
        self.tmp_batch_size = None

    def compile(self, A: Tensor) -> Tensor:

        self._A: Tensor = A
        self._W: Tensor = Tensor(self.weights)
        # biases have some broadcast risk, hope numpy auto broadcast works
        self._B: Tensor = Tensor(self.biases)

        # Z = W*A_in + B
        _Z = Matmul(self._W, self._A) + self._B

        # A_out = activation(Z)
        self.activation.build_expression(_Z)

        self._out = self.activation.expression
        return self._out

    # compute a layer's output based on the input.
    def forward(self, is_training: bool = False):

        self.tmp_batch_size = self._A.tensor.shape[1]
        self._W.tensor = self.weights
        self._B.tensor = np.repeat(self.biases, self.tmp_batch_size, axis=1)

        self._out.forward()

    # collects grad from Tensors
    def regularize_grads(self) -> np.ndarray:

        self._W_grad = self._W.grad / self.tmp_batch_size
        l2_term_for_W: np.ndarray = self.weights * self.L2_lambda # Compute regularization term
        self._W_grad += l2_term_for_W


        self._B_grad = np.sum(self._B.grad, axis=1, keepdims=True) / self.tmp_batch_size
        l2_term_for_B: np.ndarray = self.biases * self.L2_lambda # Compute regularization term
        self._B_grad += l2_term_for_B

        if self.activation.is_learnable:

            self.activation.alpha_grad = np.sum(self.activation.alpha_tensor.grad, axis=1, keepdims=True) / self.tmp_batch_size
            l2_term_for_alpha: np.ndarray = self.activation.alpha * self.L2_lambda # Compute regularization term
            self.activation.alpha_grad += l2_term_for_alpha

    def _get_weights_and_grads(self) -> list[dict]:
        params = [
            {'weight': self.weights, 'grad': self._W_grad},
            {'weight': self.biases, 'grad': self._B_grad}
        ]
        if self.activation.is_learnable:
            params.append({
                'weight': self.activation.alpha,
                'grad': self.activation.alpha_grad,
                'learnable': True,
                'constraints': self.activation.alpha_constraints
            })
        return params

    def zero_grads(self):
        # not strictly required
        # but if something breaks, you know where to find me
        if self.activation.is_learnable:
            self.activation.alpha_tensor.zero_grad()
        self._W.zero_grad()
        self._B.zero_grad()

    def _get_param_count(self) -> int:
        w = self.input_size * self.output_size
        s = self.output_size
        lp = self.output_size if self.activation.is_learnable else 0
        return w + s + lp