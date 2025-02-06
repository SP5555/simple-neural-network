import numpy as np

class Losses:
    _supported_loss = ("mae", "mse", "huber", "bce", "cce")

    # ===== Mean Absolute Error =====
    @staticmethod
    def _mae_grad(a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where(a < y, -1, 1)

    # ===== Mean Squared Error =====
    @staticmethod
    def _mse_grad(a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (a - y)

    # ===== Huber Loss =====
    @staticmethod
    def _huber_grad(a: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = 1.0 # Delta Huber Loss for future implementation
        return np.clip(a - y, -d, d)

    # ===== Binary Cross Entropy =====
    @staticmethod
    def _bce_grad(a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a) + ((1-y) / (1-a))

    # ===== Multiclass/Categorial Cross Entropy =====
    @staticmethod
    def _cce_grad(a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a)
