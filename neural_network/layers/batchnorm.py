import numpy as np
from ..auto_diff.auto_diff_reverse import Tensor, Mean, Variance, Sqrt
from ..common import PrintUtils, ParamDict, InputValidationError
from .layer import Layer

class BatchNorm(Layer):
    """
    Batch Normalization Layer
    =====
    This layer normalizes the input across the batch to stabilize training,
    improve convergence speed, and allow for deeper networks.

    During training, it computes the mean and variance of the batch, normalizes
    the input using these statistics, and applies learnable scale (`gamma`) and
    shift (`beta`) parameters. It also maintains a moving average of the mean
    and variance for use during inference.

    Parameters
    ----------
    momentum : float, optional
        Momentum for running mean and variance computation.
        Default is `0.9`.

    epsilon : float, optional
        Small constant to prevent division by zero.
        Default is `1e-5`.
    """
    def __init__(self,
                 momentum: float = 0.9,
                 epsilon: float = 1e-5
                 ):

        if momentum < 0.0:
            raise InputValidationError("Momentum can't be negative.")
        if momentum >= 1.0:
            raise InputValidationError("Momentum must be less than 1.0.")
        if momentum >= 0.95:
            PrintUtils.print_warning(f"Warning: momentum = {momentum:.3f} may cause strong \"gliding\" behavior. " +
                                     "Consider keeping it less than 0.95")
            
        if epsilon <= 1e-14 or epsilon >= 1e-3:
            raise InputValidationError("Keep epsilon between 1e-14 and 1e-3.")

        self.input_size = None
        self.neuron_count = None

        self._momentum = momentum
        self._epsilon = Tensor(epsilon, require_grad=False)

    def build(self, A: Tensor, input_size: int) -> tuple[Tensor, int]:

        self.input_size = input_size
        self.neuron_count = input_size

        # learnable parameters
        self._gamma = Tensor(np.ones((self.neuron_count, 1))) # Scale
        self._beta = Tensor(np.zeros((self.neuron_count, 1))) # Shift

        # running averages for inference
        self._running_mean = Tensor(np.zeros((self.neuron_count, 1)), require_grad=False)
        self._running_vari = Tensor(np.ones((self.neuron_count, 1)), require_grad=False)

        # mean and variance for training
        # considered as constants within a given forward pass
        # therefore, don't require gradients
        self._batch_mean = Mean(A, require_grad=False)
        self._batch_vari = Variance(A, require_grad=False)

        # flags
        self._train_flag = Tensor(0.0, require_grad=False)
        self._infer_flag = Tensor(0.0, require_grad=False)

        # ===== expression construction =====

        # just some random boolean trickery, or "switches"?
        _mean_term = self._batch_mean * self._train_flag + self._running_mean * self._infer_flag
        _vari_term = self._batch_vari * self._train_flag + self._running_vari * self._infer_flag

        _normalized = (A - _mean_term) / Sqrt(_vari_term + self._epsilon)

        self._out = self._gamma * _normalized + self._beta

        return self._out, self.neuron_count
    
    def pre_setup_tensors(self, batch_size: int, is_training = False):

        self._tmp_batch_size = batch_size

        if is_training:
            self._train_flag.assign(1.0)
            self._infer_flag.assign(0.0)
        else:
            self._train_flag.assign(0.0)
            self._infer_flag.assign(1.0)

    def post_setup_tensors(self, is_training: bool = False):

        if is_training:
            self._running_mean.assign(self._momentum * self._running_mean.tensor + (1 - self._momentum) * self._batch_mean.tensor)
            self._running_vari.assign(self._momentum * self._running_vari.tensor + (1 - self._momentum) * self._batch_vari.tensor)

    def prepare_grads(self):
        self._gamma_grad = np.sum(self._gamma.grad, axis=1, keepdims=True) / self._tmp_batch_size
        self._beta_grad = np.sum(self._beta.grad, axis=1, keepdims=True) / self._tmp_batch_size

    def _get_weights_and_grads(self) -> list[ParamDict]:
        params = [
            {'weight': self._gamma, 'grad': self._gamma_grad},
            {'weight': self._beta, 'grad': self._beta_grad}
        ]
        return params

    def zero_grads(self):
        self._gamma.zero_grad()
        self._beta.zero_grad()

    def _get_param_count(self) -> int:
        # each neuron has gamma and beta
        return self.neuron_count * 2