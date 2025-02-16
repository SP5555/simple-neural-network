import numpy as np
from ..auto_diff.auto_diff_reverse import Tensor, Sigmoid as SigOp, Tanh as TanhOp, Log, Exp
from ..auto_diff.auto_diff_reverse.operations import Operation

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
        self.Z: Tensor = None
        self.alpha_tensor: Tensor = None

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
    
    def forward(self):
        self.expression.forward()
    
    def backward(self, seed: np.ndarray):
        self.expression.backward(seed)
    
    def evaluate(self) -> np.ndarray:
        return self.expression.evaluate()

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(is_LL_multilabel_act=True)
    
    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        self.expression = SigOp(self.Z)

class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        self.expression = TanhOp(self.Z)

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        Z_p = Tensor(x >= 0)
        self.expression = self.Z * Z_p

class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        Z_p = Tensor(x >= 0)
        Z_n = Tensor(x < 0)
        self.expression = (self.Z * Z_p) + (Tensor(0.01) * self.Z * Z_n)

class PReLU(Activation):
    def __init__(self):
        super().__init__(is_learnable=True,
                         alpha_initial=0.01,
                         alpha_constraints=(0.001, 0.1))
        
    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        self.alpha_tensor = Tensor(self.alpha)
        Z_p = Tensor(x > 0)
        Z_n = Tensor(x <= 0)
        self.expression = (self.Z * Z_p) + (self.alpha_tensor * self.Z * Z_n)

class Softplus(Activation):
    def __init__(self):
        super().__init__()
    
    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        self.expression = Log(Tensor(1.0) + Exp(self.Z))

class Swish_Fixed(Activation):
    def __init__(self):
        super().__init__()

    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        self.expression = self.Z * SigOp(self.Z)

class Swish(Activation):
    def __init__(self):
        super().__init__(is_learnable=True,
                         alpha_initial=1.0,
                         alpha_constraints=(0.5, 5.0))
        
    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        self.alpha_tensor = Tensor(self.alpha)
        self.expression = self.Z * SigOp(self.alpha_tensor * self.Z)
    
class Linear(Activation):
    def __init__(self):
        super().__init__(is_LL_exclusive=True,
                         is_LL_regression_act=True)
        
    def build_expression(self, x: np.ndarray) -> None:
        self.Z = Tensor(x)
        self.expression = self.Z

class Softmax(Activation):
    # Softmax is one hell of a tricky activation
    # to get the auto-diff system to work with
    # in fact, everything is done in here without relying on auto-diff
    def __init__(self):
        super().__init__(is_LL_exclusive=True,
                         is_LL_multiclass_act=True,
                         is_dropout_incompatible=True)

    # expression.forward() would technically do nothing
    def build_expression(self, x: np.ndarray) -> np.ndarray:
        self.Z = Tensor(x)
        self.expression = self.Z

    # instead of calling expression.forward(),
    # the forwarded value is manually calculated here
    # this is due to interdependencies between outputs as
    # auto-diff system does not understand the use of np.sum and np.max
    def forward(self):
        exp = np.exp(self.Z.tensor - np.max(self.Z.tensor, axis=0, keepdims=True))
        self.expression.tensor = exp / np.sum(exp, axis=0, keepdims=True)

    # same goes here
    # since auto-diff system does not understand the use of np.sum and np.max,
    # the manual gradient calculation is offloaded to this function here.
    def backward(self, seed: np.ndarray) -> np.ndarray:
        # math
        # S_i is softmax(z_i)
        # Jacobian = diag(S) - S dot S.T
        # where each entry is
        # dS_i/dz_j = S_i * (delta_ij - S_j)
        # delta_ij is Kronecker delta term
        # (Simply put, it is an entry in Identity matrix, either 0 or 1)

        # dL/dS = seed
        # dL/dz = dL/dS dot dS/dz = Jacobian dot seed

        # But, we can avoid constructing Jacobian (which would be a sweet 3D tensor nightmare)
        # each input z's gradient dL/dz_i of dL/dz is given as follows:
        # dL/dz_i = Sum[ S_i * ( delta_ij - S_j ) * dL/dS_j ]  // j goes through all output neurons
        # dL/dz_i = S_i * Sum[ ( delta_ij - S_j ) * dL/dS_j ]  // factor out S_i
        # dL/dz_i = S_i * ( dL/dS_i - Sum[ S_j * dL/dS_j ] )   // break down delta_ij term
        # dL/dz_i = S_i * ( seed_i - Sum[ S_j * seed_j ])

        # this is softmax
        S = self.expression.tensor

        # this line took years off my lifespan
        dL_dz = S * (seed - np.sum(S * seed, axis = 0, keepdims=True))

        # this backward pass call just accumulates into partials
        # because all calculations are already done inside dL/dz term 
        self.expression.backward(dL_dz)
