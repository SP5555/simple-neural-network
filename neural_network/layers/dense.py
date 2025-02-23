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

    use_bias : bool, optional
        Whether to include a bias term (`B`) in the layer.  
        Default is `True`. If set to `False`, the layer performs `Z = W*A` without a bias shift.

    activation : Activation, optional
        Activation function applied after the linear transformation `Z = W*A + B`.\\
        Default is `None`. Without activation, the layer behaves
        as a pure linear transformation.

    weight_decay : float, optional
        L2 regularization strength for the weights.
        Default is `0.0`, meaning no regularization.
    """
    def __init__(self,
                 neuron_count: int,
                 use_bias: bool         = True,
                 activation: Activation = None,
                 weight_decay: float    = 0.0
                 ) -> None:
        
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

        self.use_bias = use_bias

        self.is_linear = True if activation is None else False
        self.activation = activation

        self.L2_lambda = weight_decay

    def build(self, A: Tensor, input_size: int) -> tuple[Tensor, int]:

        self.input_size = input_size

        # tmp vars
        self.tmp_batch_size = None

        self._W: Tensor = Tensor(np.random.randn(self.neuron_count, self.input_size) * np.sqrt(2/self.input_size))
        self._W_grad: np.ndarray = None

        if self.use_bias:
            self._B: Tensor = Tensor(np.random.randn(self.neuron_count, 1))
            self._B_grad: np.ndarray = None

        if not self.is_linear:
            self.activation.build_alpha_tensor(self.neuron_count)

        # ===== expression construction =====

        if self.use_bias:
            _Z = Matmul(self._W, A) + self._B
        else:
            _Z = Matmul(self._W, A)

        if not self.is_linear:
            self.activation.build_expression(_Z)
            self._out = self.activation.expression
        else:
            self._out = _Z
        return self._out, self.neuron_count

    def setup_tensors(self, batch_size: int, is_training: bool = False):

        self.tmp_batch_size = batch_size

    def regularize_grads(self):

        self._W_grad = self._W.grad / self.tmp_batch_size
        self._W_grad += self._W.tensor * self.L2_lambda

        if self.use_bias:

            self._B_grad = np.sum(self._B.grad, axis=1, keepdims=True) / self.tmp_batch_size
            self._B_grad += self._B.tensor * self.L2_lambda

        if not self.is_linear and self.activation.is_learnable:

            self.activation._alpha_grad = np.sum(self.activation._alpha.grad, axis=1, keepdims=True) / self.tmp_batch_size
            self.activation._alpha_grad += self.activation._alpha.tensor * self.L2_lambda

    def _get_weights_and_grads(self) -> list[ParamDict]:
        params = [
            {'weight': self._W, 'grad': self._W_grad}
        ]
        if self.use_bias:
            params.append({
                'weight': self._B, 'grad': self._B_grad
            })
        if not self.is_linear and self.activation.is_learnable:
            params.append({
                'weight': self.activation._alpha,
                'grad': self.activation._alpha_grad,
                'learnable': True,
                'constraints': self.activation._alpha_constraints
            })
        return params

    def zero_grads(self):
        self._W.zero_grad()
        if self.use_bias:
            self._B.zero_grad()
        if not self.is_linear and self.activation.is_learnable:
            self.activation._alpha.zero_grad()

    def _get_param_count(self) -> int:
        if self.input_size is None:
            raise InputValidationError("Layer has not been built yet.")
        w = self.input_size * self.neuron_count
        b = self.neuron_count if self.use_bias else 0
        lp = self.neuron_count if not self.is_linear and self.activation.is_learnable else 0
        return w + b + lp