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
        self._A_in = None   # shape: (input_size, batch_size)
        self._W    = None   # shape: (output_size, input_size)
        self._B    = None   # shape: (output_size, 1) broadcast to (output_size, batch_size)
        self._Z    = None   # shape: (output_size, batch_size)

    # compute a layer's output based on the input.
    def forward(self, input: np.ndarray, is_training: bool = False) -> np.ndarray:

        self._A_in: Tensor = Tensor(input)
        self._W: Tensor = Tensor(self.weights)
        self._B: Tensor = Tensor(np.repeat(self.biases, input.shape[1], axis=1)) # broadcast

        # Z = W*A_in + B
        self._Z = Matmul(self._W, self._A_in) + self._B

        # A_out = activation(Z)
        self.activation.build_expression(self._Z)
        self.activation.forward()
        return self.activation.evaluate()

    # basically BACKPROPAGATION
    # returns the input gradient which is required for the previous layer's backward pass.
    def backward(self, seed: np.ndarray) -> np.ndarray:

        batch_size = seed.shape[1]

        # auto diff reverse mode backward call
        # situates all tensors with their gradients
        self.activation.backward(seed)

        # Math
        # Z = W*A_in + B
        # A = activation(Z, learn_b) # learnable parem is used only in some activations

        # CALCULATE derivative of loss with respect to weights
        # dL/dW
        # = dL/dA * dA/dZ * dZ/dW
        # = dL/dA * dA/dZ * A_in
        self._W_grad = self._W.grad / batch_size
        l2_term_for_W: np.ndarray = self.weights * self.L2_lambda # Compute regularization term
        self._W_grad += l2_term_for_W

        # CALCULATE derivative of loss with respect to biases
        # dL/db(n)
        # = dL/dA * dA/dZ * dZ/dB
        # = dL/dA * dA/dZ * 1
        self._B_grad = np.sum(self._B.grad, axis=1, keepdims=True) / batch_size
        l2_term_for_B: np.ndarray = self.biases * self.L2_lambda # Compute regularization term
        self._B_grad += l2_term_for_B

        if self.activation.is_learnable:
            # CALCULATE derivative of loss with respect to learnable parameter
            # dL/dlearn_b
            # = dL/dA * dA/dlearn_b
            self.activation.alpha_grad = np.sum(self.activation.alpha_tensor.grad, axis=1, keepdims=True) / batch_size
            l2_term_for_alpha: np.ndarray = self.activation.alpha * self.L2_lambda # Compute regularization term
            self.activation.alpha_grad += l2_term_for_alpha

            # not strictly required
            self.activation.alpha_tensor.zero_grad()

        # first layer does not have previous layer
        # not need to return gradient of loss
        if self._is_first: return

        # "seed" or gradient of loss for previous layer
        # NOTE: A_in affects all A, so backpropagation to A_in will be related to all A
        # dL/dA_in
        # = dL/dA * dA/dZ * dZ/dA_in
        # = dL/dA * dA/dZ * W
        seed = self._A_in.grad

        # not strictly required
        # but if something breaks, you know where to find me
        self._A_in.zero_grad()
        self._W.zero_grad()
        self._B.zero_grad()

        return seed

    def _get_params(self) -> list[dict]:
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
    
    def _get_param_count(self) -> int:
        w = self.input_size * self.output_size
        s = self.output_size
        lp = self.output_size if self.activation.is_learnable else 0
        return w + s + lp