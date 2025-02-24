import numpy as np
from ..auto_diff.auto_diff_reverse import Tensor, Mean, Variance, Sqrt
from ..common import PrintUtils, ParamDict, InputValidationError
from .layer import Layer

class BatchNorm(Layer):
    """
    Batch Normalization Layer.

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

        self.input_size = None
        self.neuron_count = None

        self.momentum = momentum
        self.epsilon = Tensor(epsilon, require_grad=False)

    def build(self, A: Tensor, input_size: int) -> tuple[Tensor, int]:

        self.input_size = input_size
        self.neuron_count = input_size

        # learnable parameters
        self._gamma = Tensor(np.ones((self.neuron_count, 1))) # Scale
        self._beta = Tensor(np.zeros((self.neuron_count, 1))) # Shift

        # running averages for inference
        self.running_mean = Tensor(np.zeros((self.neuron_count, 1)), require_grad=False)
        self.running_vari = Tensor(np.ones((self.neuron_count, 1)), require_grad=False)

        # mean and variance for training
        self.batch_mean = Mean(A)
        self.batch_vari = Variance(A)

        # flags
        self.train_flag = Tensor(0.0, require_grad=False)
        self.infer_flag = Tensor(0.0, require_grad=False)

        # just some random boolean trickery, or "switches"?
        mean_term = self.batch_mean * self.train_flag + self.running_mean * self.infer_flag
        vari_term = self.batch_vari * self.train_flag + self.running_vari * self.infer_flag

        normalized = (A - mean_term) / Sqrt(vari_term + self.epsilon)

        self._out = self._gamma * normalized + self._beta

        return self._out, self.neuron_count
    
    def setup_tensors(self, batch_size: int, is_training = False):
        
        self.tmp_batch_size = batch_size

        if is_training:
            self.train_flag.assign(1.0)
            self.infer_flag.assign(0.0)
        else:
            self.train_flag.assign(0.0)
            self.infer_flag.assign(1.0)

    def sync_after_backward(self, is_training: bool = False):

        if is_training:
            self.running_mean.assign(self.momentum * self.running_mean.tensor + (1 - self.momentum) * self.batch_mean.tensor)
            self.running_vari.assign(self.momentum * self.running_vari.tensor + (1 - self.momentum) * self.batch_vari.tensor)

    def prepare_grads(self):
        self._gamma_grad = np.sum(self._gamma.grad, axis=1, keepdims=True) / self.tmp_batch_size
        self._beta_grad = np.sum(self._beta.grad, axis=1, keepdims=True) / self.tmp_batch_size

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