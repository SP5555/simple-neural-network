import numpy as np
from ..auto_diff.auto_diff_reverse import Tensor
from ..common import PrintUtils, ParamDict, InputValidationError
from .layer import Layer

class Dropout(Layer):
    """
    A dropout layer that randomly drops neurons during training to prevent overfitting.

    Parameters
    ----------
    dropout_rate : float
        Fraction of neurons to drop during training. Should be between `0.0`
        (no dropout) and `1.0` (drop everything, never do this).
        Typical values are between `0.2` and `0.5`.

    batch_wise : bool, optional
        If `True`, the same dropout mask is applied across the entire batch.
        If `False`, dropout is applied independently to each sample.
        Default is `False`.
    """
    def __init__(self,
                 dropout_rate: float,
                 batch_wise: bool = False) -> None:

        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise InputValidationError("Dropout probability must be between 0.0 and 1.0")
        if dropout_rate > 0.5:
            PrintUtils.print_warning(f"Dropout Probability of {dropout_rate} is too high. Consider less than 0.5")

        self.input_size = None
        self.neuron_count = None

        self.dp = dropout_rate
        self.batch_wise = batch_wise
    
    def build(self, A: Tensor, input_size: int, is_first: bool = False, is_final: bool = False) -> tuple[Tensor, int]:
        if is_final:
            PrintUtils.print_warning("Using a dropout layer as the final layer is not recommended.")
        # if self.activation.is_dropout_incompatible:
        #     raise InputValidationError(f"{self.activation.__class__.__name__} is not compatible in the dropout layer.")

        self.input_size = input_size
        self.neuron_count = input_size

        # tmp vars
        self.tmp_batch_size = None

        # auto-diff tensor objects
        self.mask = Tensor(1.0, require_grad=False)
        self.rescaler = Tensor(1.0, require_grad=False)

        # ===== expression construction =====

        # Apply dropout
        # mask    : zero out dp fraction of activations
        # rescaler: scale up the surviving activations
        self._out = A * self.mask * self.rescaler
        return self._out, self.neuron_count

    def setup_tensors(self, batch_size: int, is_training: bool = False):

        self.tmp_batch_size = batch_size

        if is_training:
            # standard dropout  : randomly drops neurons individually within each sample
            # batch-wise dropout: same dropout pattern to all neurons within a mini-batch
            shape = (self.neuron_count, self.tmp_batch_size)
            if self.batch_wise:
                shape = (self.neuron_count, 1)
            # create a mask where a neuron has a 1-dp chance to remain active
            self.mask.assign(np.random.binomial(n=1, p=1-self.dp, size=shape))
            self.rescaler.assign(1.0 / (1.0 - self.dp))
        else:
            self.mask.assign(1.0)
            self.rescaler.assign(1.0)

    # dropout layer has no learnable parameters
    def regularize_grads(self):
        pass
    
    def _get_weights_and_grads(self) -> list[ParamDict]:
        return []

    def zero_grads(self):
        pass

    def _get_param_count(self) -> int:
        return 0