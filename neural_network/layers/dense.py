import numpy as np
from ..activations.activation import Activation
from ..auto_diff.auto_diff_reverse import Tensor, Matmul
from ..common import PrintUtils, ParamDict, InputValidationError
from .layer import Layer

class Dense(Layer):
    """
    A fully connected (dense) layer where every input neuron 
    is connected to every output neuron.

    Parameters
    ----------
    neuron_count : int
        Number of neurons in this layer.
    
    activation : Activation
        Activation function of the neurons.
    
    weight_decay : float, optional
        Strength of L2 regularization.
        Default is `0.0`, meaning no regularization.
    """
    def __init__(self,
                 neuron_count: int,
                 activation: Activation,
                 weight_decay: float = 0.0) -> None:
        
        super().__init__()

        if neuron_count == 0:
            raise InputValidationError("A layer can't have 0 output (0 neurons).")
        
        # L2 Regularization Strength
        # low reg strength -> cook in class, and fail in exam; overfit
        # high reg strength -> I am dumb dumb, can't learn; underfit
        # Large weights and biases will are penalized more aggressively than small ones
        # Don't set it too large, at most 0.01 (unless you know what you're doing)
        #     regularized_loss     = parameter_los      + 1/2 * L2_lambda * parameter^2
        #     regularized_gradient = parameter_gradient +       L2_lambda * parameter
        if weight_decay < 0.0:
            raise InputValidationError("Regularization Strength can't be negative.")
        if weight_decay > 0.01:
            PrintUtils.print_warning(f"Warning: Regularization Strength {weight_decay:.3f} is strong. Consider keeping it less than 0.01")

        self.input_size = None
        self.neuron_count = neuron_count

        self.activation = activation

        self.L2_lambda = weight_decay

    def build(self, A: Tensor, input_size: int, is_first: bool = False, is_final: bool = False) -> tuple[Tensor, int]:

        self.input_size = input_size

        # helpers, they save a lot of power
        self._is_first = is_first
        self._is_final = is_final

        if not self._is_final and self.activation.is_LL_exclusive:
            raise InputValidationError(f"{self.activation.__class__.__name__} activation can't be used in hidden layers.")

        # auto-diff tensor objects
        self._W: Tensor = Tensor(np.random.randn(self.neuron_count, self.input_size) * np.sqrt(2/self.input_size))
        self._B: Tensor = Tensor(np.random.randn(self.neuron_count, 1))
        self.activation.build_alpha_tensor(self.neuron_count)
 
        # gradient container numpy array objects
        self._W_grad = None
        self._B_grad = None

        # tmp vars
        self.tmp_batch_size = None

        # Z = W*A_in + B
        _Z = Matmul(self._W, A) + self._B

        # A_out = activation(Z)
        self.activation.build_expression(_Z)

        self._out = self.activation.expression
        return self._out, self.neuron_count

    def setup_tensors(self, batch_size: int, is_training: bool = False):

        self.tmp_batch_size = batch_size

    def regularize_grads(self):

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
        if self.input_size is None:
            raise InputValidationError("Layer has not been built yet.")
        w = self.input_size * self.neuron_count
        s = self.neuron_count
        lp = self.neuron_count if self.activation.is_learnable else 0
        return w + s + lp