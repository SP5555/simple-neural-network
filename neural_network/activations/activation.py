import numpy as np
from ..auto_diff.auto_diff_reverse import (
    Tensor,
    Sigmoid as SigAD,
    Tanh as TanhAD,
    Log,
    Exp,
    Softmax as SoftmaxAD,
    Maximum,
    Minimum
)
from ..auto_diff.auto_diff_reverse.operations import Operation

class Activation:
    """
    Abstract base class for all activation functions.

    This class serves as a template and cannot be used directly in the models. \\
    Methods in this class raise `NotImplementedError` to enforce implementation 
    in derived child classes.
    """
    def __init__(self,
                 is_LL_regression_act = False,
                 is_LL_multilabel_act = False,
                 is_LL_multiclass_act = False,
                 is_learnable = False,
                 alpha_initial: float = None,
                 alpha_constraints: tuple = None):
        self.is_LL_regression_act    = is_LL_regression_act
        self.is_LL_multilabel_act    = is_LL_multilabel_act
        self.is_LL_multiclass_act    = is_LL_multiclass_act
        self.is_learnable            = is_learnable

        # alphas are learnable parameters
        # we're running out of Greek alphabets
        self._alpha_initial: float = alpha_initial
        self._alpha_constraints: tuple = alpha_constraints
        self._alpha_grad: np.ndarray = None

        # tensor/operation auto-diff objects
        self._alpha: Tensor = None
        self.expression: Operation = None

    def build_alpha_tensor(self, output_size: int):
        if self.is_learnable:
            self._alpha = Tensor(np.full((output_size, 1), self._alpha_initial))
    
    # builds expression using auto diff tensors
    def build_expression(self, Z: Tensor):
        raise NotImplementedError
    
    def forward(self):
        self.expression.forward()
    
    def backward(self, seed: np.ndarray):
        self.expression.backward(seed)
    
    def evaluate(self) -> np.ndarray:
        return self.expression.evaluate()

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(is_LL_multilabel_act=True)
    
    def build_expression(self, Z: Tensor):
        self.expression = SigAD(Z)

class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, Z: Tensor):
        self.expression = TanhAD(Z)

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, Z: Tensor):
        zero = Tensor(0.0, requires_grad=False)
        self.expression = Maximum(Z, zero)

class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, Z: Tensor):
        zero = Tensor(0.0, requires_grad=False)
        neg_slope = Tensor(0.01, requires_grad=False)
        Z_p = Maximum(Z, zero)
        Z_n = Minimum(Z, zero)
        self.expression = Z_p + (neg_slope * Z_n)  # Leaky ReLU formula

class PReLU(Activation):
    def __init__(self):
        super().__init__(is_learnable=True,
                         alpha_initial=0.01,
                         alpha_constraints=(0.001, 0.1))

    def build_expression(self, Z: Tensor):
        zero = Tensor(0.0, requires_grad=False)
        Z_p = Maximum(Z, zero)
        Z_n = Minimum(Z, zero)
        self.expression = Z_p + (self._alpha * Z_n)

class Softplus(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, Z: Tensor):
        one = Tensor(1.0, requires_grad=False)
        self.expression = Log(one + Exp(Z))

class Swish_Fixed(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, Z: Tensor):
        self.expression = Z * SigAD(Z)

class Swish(Activation):
    def __init__(self):
        super().__init__(is_learnable=True,
                         alpha_initial=1.0,
                         alpha_constraints=(0.5, 5.0))

    def build_expression(self, Z: Tensor):
        self.expression = Z * SigAD(self._alpha * Z)
    
class Linear(Activation):
    def __init__(self):
        super().__init__(is_LL_regression_act=True)

    def build_expression(self, Z: Tensor):
        self.expression = Z

class Softmax(Activation):
    def __init__(self):
        super().__init__(is_LL_multiclass_act=True)

    def build_expression(self, Z: Tensor):
        self.expression = SoftmaxAD(Z)