import numpy as np
from ..auto_diff.auto_diff import Tensor, Sigmoid as SigOp, Tanh as TanhOp
from ..auto_diff.auto_diff.operations import Operation

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # s(x) = (tanh(x/2) + 1) / 2
    return (np.tanh(x / 2) + 1) / 2

class Activation:
    """
    Abstract base class for all activation functions.

    This class serves as a template and cannot be used directly in the models. \\
    Methods in this class raise `NotImplementedError` to enforce implementation 
    in derived child classes.
    """
    def __init__(self,
                 is_LL_exclusive = False,
                 is_LL_regression_act = False,
                 is_LL_multilabel_act = False,
                 is_LL_multiclass_act = False,
                 is_learnable = False,
                 is_dropout_incompatible = False,
                 alpha_initial: float = None,
                 alpha_constraints: tuple = None):
        self.is_LL_exclusive         = is_LL_exclusive
        self.is_LL_regression_act    = is_LL_regression_act
        self.is_LL_multilabel_act    = is_LL_multilabel_act
        self.is_LL_multiclass_act    = is_LL_multiclass_act
        self.is_learnable            = is_learnable
        self.is_dropout_incompatible = is_dropout_incompatible

        # alphas are learnable parameters
        # we're running out of Greek alphabets
        self.alpha_initial: float = alpha_initial
        self.alpha_constraints: tuple = alpha_constraints
        self.alpha: np.ndarray = None
        self.alpha_grad: np.ndarray = None
        self.expression: Operation = None

    def build_parameters(self, output_size: int) -> None:
        if self.is_learnable:
            self.alpha = np.full((output_size, 1), self.alpha_initial)
    
    # builds expression using auto diff tensors
    def build_expression(self, x: np.ndarray) -> None:
        raise NotImplementedError
    
    def forward(self) -> np.ndarray:
        return self.expression.forward()
    
    def backward(self) -> np.ndarray:
        return self.expression.backward("Z")
    
    # ===== only for learnable activations =====
    def get_param_grad(self) -> np.ndarray:
        return self.expression.backward("alpha")

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(is_LL_multilabel_act=True)
    
    def build_expression(self, x: np.ndarray) -> None:
        self.expression = SigOp(Tensor(x, "Z"))

class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, x: np.ndarray) -> None:
        self.expression = TanhOp(Tensor(x, "Z"))

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, x: np.ndarray) -> None:
        Z = Tensor(x, "Z")
        Z_p = Tensor(x >= 0)
        self.expression = Z * Z_p

class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, x: np.ndarray) -> None:
        Z = Tensor(x, "Z")
        Z_p = Tensor(x >= 0)
        Z_n = Tensor(x < 0)
        self.expression = (Z * Z_p) + (Tensor(0.01) * Z * Z_n)

class PReLU(Activation):
    def __init__(self):
        super().__init__(is_learnable=True,
                         alpha_initial=0.01,
                         alpha_constraints=(0.001, 0.1))
        
    def build_expression(self, x: np.ndarray) -> None:
        Z = Tensor(x, "Z")
        Z_p = Tensor(x > 0)
        Z_n = Tensor(x <= 0)
        self.expression = (Z * Z_p) + (Tensor(self.alpha, "alpha") * Z * Z_n)

class Swish_Fixed(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, x: np.ndarray) -> None:
        Z = Tensor(x, "Z")
        self.expression = Z * SigOp(Z)

class Swish(Activation):
    def __init__(self):
        super().__init__(is_learnable=True,
                         alpha_initial=1.0,
                         alpha_constraints=(0.5, 5.0))
        
    def build_expression(self, x: np.ndarray) -> None:
        Z = Tensor(x, "Z")
        alpha = Tensor(self.alpha, "alpha")
        self.expression = Z * SigOp(alpha * Z)
    
class Linear(Activation):
    def __init__(self):
        super().__init__(is_LL_exclusive=True,
                         is_LL_regression_act=True)
        
    def build_expression(self, x: np.ndarray) -> None:
        self.expression = Tensor(x, "Z")

class Softmax(Activation):
    def __init__(self):
        super().__init__(is_LL_exclusive=True,
                         is_LL_multiclass_act=True,
                         is_dropout_incompatible=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    # due to the nature of how softmax derivative is applied in backpropagation,
    # this function returns "stacks" of jacobian matrices.
    def backward(self, x: np.ndarray) -> np.ndarray:
        dim = x.shape[0]
        softmax = self.forward(x) # Shape: (dim, batch_size)

        softmax_expanded = softmax.T[:, :, None]  # Shape: (batch_size, dim, 1)

        # Shape: (batch_size, dim, dim)
        # matmul perform matrix multiplication on the last two dimensions here
        # each sample "slice" on 0-th axis is: I * M_softmax(dim, 1) - np.dot(M_softmax, M_softmax.T)
        jacobians = np.eye(dim)[None, :, :] * softmax_expanded - np.matmul(softmax_expanded, softmax_expanded.transpose(0, 2, 1))

        return jacobians # Shape: (batch_size, dim, dim)
