import numpy as np

class ActivationWrapper:
    def __init__(self, func: callable, name: str) -> None:
        self.func = func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class Activations:
    _supported_acts = (
        "relu", "leaky_relu", "prelu", "tanh",
        "sigmoid", "swish", "swish_f", "id",
        "linear", "softmax"
    )
    
    # only usable in last layer
    _LL_exclusive = ("id", "linear", "softmax")

    # last layer check for accuracy calculation
    _LL_regression_acts = ("id", "linear",)
    _LL_multilabel_acts = ("sigmoid",)
    _LL_multiclass_acts = ("softmax",)

    # activations with learnable parameters
    _learnable_acts = ("prelu", "swish")

    # activations that are not compatible in dropout layer
    _dropout_incomp_acts = ("softmax")
    
    # learnable parameter value dictionary
    _learn_param_values = {
        # name: (initial, low_cap, high_cap)
        "prelu": (0.01, 0.001, 0.1),
        "swish": (1.0, 0.5, 5.0)
    }
    
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
        return np.where(x > 0, x, 0.01 * x)

    @staticmethod
    def _leaky_relu_deriv(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.01)
    
    # ===== Parametric ReLU (Learnable Leaky ReLU) =====
    @staticmethod
    def _prelu(x: np.ndarray, alp: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, alp * x)

    @staticmethod
    def _prelu_deriv(x: np.ndarray, alp: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, alp)

    @staticmethod
    def _prelu_param_deriv(x: np.ndarray, alp: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 0, x)
    
    # ===== Swish with disabled learnable parameter =====
    @staticmethod
    def _swish_fixed(x: np.ndarray) -> np.ndarray:
        # swish(x) = x * s(x)
        return x * Activations._sigmoid(x)
    
    @staticmethod
    def _swish_fixed_deriv(x: np.ndarray) -> np.ndarray:
        # dswish(x)/dx = s(x) * (1 + x * (1 - s(x)))
        s = Activations._sigmoid(x)
        return s * (1 + x * (1 - s))

    # ===== Swish with learnable parameter =====
    @staticmethod
    def _swish(x: np.ndarray, alp: np.ndarray) -> np.ndarray:
        # swish(x, b) = x * s(bx)
        s = Activations._sigmoid(alp * x)
        return x * s
    
    @staticmethod
    def _swish_deriv(x: np.ndarray, alp: np.ndarray) -> np.ndarray:
        # dswish(x, b)/dx = s(bx) * (1 + bx * (1 - s(bx)))
        alpx = alp * x
        s = Activations._sigmoid(alpx)
        return s * (1 + alpx * (1 - s))

    @staticmethod
    def _swish_param_deriv(x: np.ndarray, alp:np.ndarray) -> np.ndarray:
        # dswish(x, b)/db = x^2 * s(bx) * (1 - s(bx))
        s = Activations._sigmoid(alp * x)
        return x * x * s * (1 - s)

    # ===== Linear (Identity) =====
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
