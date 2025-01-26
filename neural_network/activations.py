import numpy as np

# ACTIVATION FUNCTIONS
class Activations:
    # ===== Sigmoid =====
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # s(x) = (tanh(x/2) + 1) / 2
        return (np.tanh(x / 2) + 1) / 2

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        # s'(x) = s(x) * (1 - s(x))
        s: np.ndarray = (np.tanh(x / 2) + 1) / 2
        return s*(1-s)

    # ===== Tanh =====
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        # tanh'(x) = 1 - tanh(x)^2
        t = np.tanh(x)
        return 1 - t*t

    # ===== ReLU =====
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    # ===== Leaky ReLU =====
    @staticmethod
    def leaky_relu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.1 * x)

    @staticmethod
    def leaky_relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.1)

    # ===== Linear Activation =====
    @staticmethod
    def id(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def id_derivative(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    # ===== Softmax =====
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    @staticmethod
    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        dim = x.shape[0]
        softmax = Activations.softmax(x) # Shape: (dim, batch_size)

        softmax_expanded = softmax.T[:, :, None]  # Shape: (batch_size, dim, 1)

        # Shape: (batch_size, dim, dim)
        # matmul perform matrix multiplication on the last two dimensions here
        # each sample "slice" on 0-th axis is: I * M_softmax(dim, 1) - np.dot(M_softmax, M_softmax.T)
        jacobians = np.eye(dim)[None, :, :] * softmax_expanded - np.matmul(softmax_expanded, softmax_expanded.transpose(0, 2, 1))

        return jacobians # Shape: (batch_size, dim, dim)
