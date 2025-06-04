import numpy as np
from ..activations.activation import Activation
from ..auto_diff.auto_diff_reverse import Tensor, Matmul
from ..common import ParamDict, InputValidationError
from .regularizablelayer import RegularizableLayer

class Dense(RegularizableLayer):
    """
    Dense Layer
    =====
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
        If not defined, it will use the global `weight_decay` value 
        defined in the `NeuralNetwork` class level.
    """
    def __init__(self,
                 neuron_count: int,
                 use_bias: bool             = True,
                 activation: Activation     = None,
                 weight_decay: float | None = None
                 ):
        
        if neuron_count == 0:
            raise InputValidationError("A layer can't have 0 output (0 neurons).")

        super().__init__(weight_decay=weight_decay)

        self.input_size = None
        self.neuron_count = neuron_count

        self._use_bias = use_bias

        self._is_linear = True if activation is None else False
        self._activation = activation

    def build(self, A: Tensor, input_size: int, weight_decay: float | None = None) -> tuple[Tensor, int]:

        self.input_size = input_size

        # L2 lambda override
        self._L2_lambda = (
            self.weight_decay if self.weight_decay is not None  # layer-specific
            else weight_decay if weight_decay is not None       # global
            else 0.0
        )

        # tmp vars
        self._tmp_batch_size = None

        self._W: Tensor = Tensor(np.random.randn(self.neuron_count, self.input_size) * np.sqrt(2/self.input_size))
        self._W_grad: np.ndarray = None

        if self._use_bias:
            self._B: Tensor = Tensor(np.random.randn(self.neuron_count, 1))
            self._B_grad: np.ndarray = None

        if not self._is_linear:
            self._activation.build_alpha_tensor(self.neuron_count)

        # ===== expression construction =====

        if self._use_bias:
            _Z = Matmul(self._W, A) + self._B
        else:
            _Z = Matmul(self._W, A)

        if not self._is_linear:
            self._activation.build_expression(_Z)
            self._out = self._activation.expression
        else:
            self._out = _Z
        return self._out, self.neuron_count

    def pre_setup_tensors(self, batch_size: int, is_training: bool = False):

        self._tmp_batch_size = batch_size

    def post_setup_tensors(self, is_training: bool = False):
        pass

    def prepare_grads(self):

        self._W_grad = self._W.grad / self._tmp_batch_size
        self._W_grad += self._W.tensor * self._L2_lambda

        if self._use_bias:

            self._B_grad = np.sum(self._B.grad, axis=1, keepdims=True) / self._tmp_batch_size

        if not self._is_linear and self._activation.is_learnable:

            self._activation._alpha_grad = np.sum(self._activation._alpha.grad, axis=1, keepdims=True) / self._tmp_batch_size
            self._activation._alpha_grad += self._activation._alpha.tensor * self._L2_lambda

    def _get_weights_and_grads(self) -> list[ParamDict]:
        params = [
            {'weight': self._W, 'grad': self._W_grad}
        ]
        if self._use_bias:
            params.append({
                'weight': self._B, 'grad': self._B_grad
            })
        if not self._is_linear and self._activation.is_learnable:
            params.append({
                'weight': self._activation._alpha,
                'grad': self._activation._alpha_grad,
                'learnable': True,
                'constraints': self._activation._alpha_constraints
            })
        return params

    def zero_grads(self):
        self._W.zero_grad()
        if self._use_bias:
            self._B.zero_grad()
        if not self._is_linear and self._activation.is_learnable:
            self._activation._alpha.zero_grad()

    def _get_param_count(self) -> int:
        if self.input_size is None:
            raise InputValidationError("Layer has not been built yet.")
        w = self.input_size * self.neuron_count
        b = self.neuron_count if self._use_bias else 0
        lp = self.neuron_count if not self._is_linear and self._activation.is_learnable else 0
        return w + b + lp