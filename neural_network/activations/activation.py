import numpy as np

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

    def build_parameters(self, output_size: int) -> None:
        if self.is_learnable:
            self.alpha = np.full((output_size, 1), self.alpha_initial)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    # ===== only for learnable activations =====
    def get_param_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the activation function w.r.t. its learnable parameter alpha.
        Used for updating alpha during training.
        """
        raise NotImplementedError

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(is_LL_multilabel_act=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        # s'(x) = s(x) * (1 - s(x))
        s: np.ndarray = _sigmoid(x)
        return s*(1-s)

class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        # tanh'(x) = 1 - tanh(x)^2
        t = np.tanh(x)
        return 1 - t*t

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.01 * x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.01)

class PReLU(Activation):
    def __init__(self):
        super().__init__(is_learnable=True,
                         alpha_initial=0.01,
                         alpha_constraints=(0.001, 0.1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)

    def get_param_grad(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 0, x)

class Swish_Fixed(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # swish(x) = x * s(x)
        s = _sigmoid(x)
        return x * s
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        # dswish(x)/dx = s(x) * (1 + x * (1 - s(x)))
        s = _sigmoid(x)
        return s * (1 + x * (1 - s))

class Swish(Activation):
    def __init__(self):
        super().__init__(is_learnable=True,
                         alpha_initial=1.0,
                         alpha_constraints=(0.5, 5.0))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # swish(x, b) = x * s(bx)
        s = _sigmoid(self.alpha * x)
        return x * s

    def backward(self, x: np.ndarray) -> np.ndarray:
        # dswish(x, b)/dx = s(bx) * (1 + bx * (1 - s(bx)))
        alpx = self.alpha * x
        s = _sigmoid(alpx)
        return s * (1 + alpx * (1 - s))

    def get_param_grad(self, x: np.ndarray) -> np.ndarray:
        # dswish(x, b)/db = x^2 * s(bx) * (1 - s(bx))
        s = _sigmoid(self.alpha * x)
        return x * x * s * (1 - s)
    
class Linear(Activation):
    def __init__(self):
        super().__init__(is_LL_exclusive=True,
                         is_LL_regression_act=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

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
