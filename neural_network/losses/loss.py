from ..exceptions import InputValidationError
import numpy as np

class Loss:
    """
    Abstract base class for all loss functions.

    This class serves as a template and cannot be used directly in the models. \\
    Methods in this class raise `NotImplementedError` to enforce implementation 
    in derived child classes.
    """
    def grad(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# ===== Mean Absolute Error =====
class MAE(Loss):
    def grad(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(a < y, -1, 1)

# ===== Mean Squared Error =====
class MSE(Loss):
    def grad(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (a - y)

# ===== Huber Loss =====
class Huber(Loss):
    def __init__(self, delta: float):
        if delta <= 0.0:
            raise InputValidationError("Huber constant delta must be positive.")
        self.d = delta

    def grad(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.clip(a - y, -self.d, self.d)

# ===== Binary Cross Entropy =====
class BCE(Loss):
    def grad(self, a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a) + ((1-y) / (1-a))

# ===== Multiclass/Categorial Cross Entropy =====
class CCE(Loss):
    def grad(self, a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a)
