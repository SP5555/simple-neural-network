import numpy as np

# ACTIVATION FUNCTIONS
class Activations:
    # ===== Sigmoid =====
    @staticmethod
    def sigmoid(np_array: np.ndarray) -> np.ndarray:
        # s(x) = (tanh(x/2) + 1) / 2
        return (np.tanh(np_array / 2) + 1) / 2

    @staticmethod
    def sigmoid_derivative(np_array: np.ndarray) -> np.ndarray:
        # s'(x) = s(x) * (1 - s(x))
        s: np.ndarray = (np.tanh(np_array / 2) + 1) / 2
        return s*(1-s)

    # ===== Tanh =====
    @staticmethod
    def tanh(np_array: np.ndarray) -> np.ndarray:
        return np.tanh(np_array)

    @staticmethod
    def tanh_derivative(np_array: np.ndarray) -> np.ndarray:
        # tanh'(x) = 1 - tanh(x)^2
        t = np.tanh(np_array)
        return 1 - t*t

    # ===== ReLU =====
    @staticmethod
    def relu(np_array: np.ndarray) -> np.ndarray:
        return np.maximum(0, np_array)

    @staticmethod
    def relu_derivative(np_array: np.ndarray) -> np.ndarray:
        return np.where(np_array > 0, 1, 0)

    # ===== Leaky ReLU =====
    @staticmethod
    def leaky_relu(np_array: np.ndarray) -> np.ndarray:
        return np.where(np_array > 0, np_array, 0.1 * np_array)

    @staticmethod
    def leaky_relu_derivative(np_array: np.ndarray) -> np.ndarray:
        return np.where(np_array > 0, 1, 0.1)

    # ===== Linear Activation =====
    @staticmethod
    def id(np_array: np.ndarray) -> np.ndarray:
        return np_array

    @staticmethod
    def id_derivative(np_array: np.ndarray) -> np.ndarray:
        return np.ones_like(np_array)

    # ===== Softmax =====
    @staticmethod
    def softmax(np_array: np.ndarray) -> np.ndarray:
        exp = np.exp(np_array - np.max(np_array, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    @staticmethod
    def softmax_derivative(np_array: np.ndarray) -> np.ndarray:
        dim = np_array.shape[0]
        softmax = Activations.softmax(np_array) # Shape: (dim, batch_size)

        softmax_expanded = softmax.T[:, :, None]  # Shape: (batch_size, dim, 1)

        # Shape: (batch_size, dim, dim)
        # matmul perform matrix multiplication on the last two dimensions here
        # each sample "slice" on 0-th axis is: I * M_softmax(dim, 1) - np.dot(M_softmax, M_softmax.T)
        jacobians = np.eye(dim)[None, :, :] * softmax_expanded - np.matmul(softmax_expanded, softmax_expanded.transpose(0, 2, 1))

        return jacobians # Shape: (batch_size, dim, dim)
