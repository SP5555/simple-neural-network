import numpy as np

class ActivationWrapper:
    def __init__(self, func: callable, name: str) -> None:
        self.func = func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# ACTIVATION FUNCTIONS
class Activations:
    _LL_exclusive = ("id", "linear", "softmax")
    _classification_LL_acts = ("sigmoid", "tanh", "softmax")
    _learnable_acts = ()
    
    # ===== Sigmoid =====
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # s(x) = (tanh(x/2) + 1) / 2
        return (np.tanh(x / 2) + 1) / 2

    @staticmethod
    def _sigmoid_deriv(x: np.ndarray) -> np.ndarray:
        # s'(x) = s(x) * (1 - s(x))
        s: np.ndarray = (np.tanh(x / 2) + 1) / 2
        return s*(1-s)

    # ===== Tanh =====
    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def _tanh_deriv(x: np.ndarray) -> np.ndarray:
        # tanh'(x) = 1 - tanh(x)^2
        t = np.tanh(x)
        return 1 - t*t

    # ===== ReLU =====
    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _relu_deriv(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    # ===== Leaky ReLU =====
    @staticmethod
    def _leaky_relu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.1 * x)

    @staticmethod
    def _leaky_relu_deriv(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.1)

    # ===== Swish ===== learnable parameter
    @staticmethod
    def _swish(x: np.ndarray, b: np.float64) -> np.ndarray:
        # swish(x, b) = x * s(bx)
        s = Activations._sigmoid(b * x)
        return x * s
    
    @staticmethod
    def _swish_deriv(x: np.ndarray, b: np.float64) -> np.ndarray:
        # dswish(x, b)/dx = s(bx) * (1 + bx * (1 - s(bx)))
        bx = b * x
        s = Activations._sigmoid(bx)
        return s * (1 + bx * (1 - s))

    @staticmethod
    def _swish_learnable_deriv(x: np.ndarray, b:np.float64) -> np.ndarray:
        # dswish(x, b)/db = x^2 * s(bx) * (1 - s(bx))
        s = Activations._sigmoid(b * x)
        return x * x * s * (1 - s)

    # ===== Linear Activation =====
    @staticmethod
    def _id(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def _id_deriv(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    # ===== Softmax =====
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    @staticmethod
    # due to the nature of how softmax derivative is applied in backpropagation,
    # this function returns "stacks" of jacobian matrices.
    def _softmax_deriv(x: np.ndarray) -> np.ndarray:
        dim = x.shape[0]
        softmax = Activations._softmax(x) # Shape: (dim, batch_size)

        softmax_expanded = softmax.T[:, :, None]  # Shape: (batch_size, dim, 1)

        # Shape: (batch_size, dim, dim)
        # matmul perform matrix multiplication on the last two dimensions here
        # each sample "slice" on 0-th axis is: I * M_softmax(dim, 1) - np.dot(M_softmax, M_softmax.T)
        jacobians = np.eye(dim)[None, :, :] * softmax_expanded - np.matmul(softmax_expanded, softmax_expanded.transpose(0, 2, 1))

        return jacobians # Shape: (batch_size, dim, dim)
