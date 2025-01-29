import numpy as np

class Losses:
    _supported_loss = ("mse", "bce", "cce")

    # ===== Mean Squared Error =====
    @staticmethod
    def _mse_grad(a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (a - y)
    
    # ===== Binary Cross Entropy =====
    @staticmethod
    def _bce_grad(a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a) + ((1-y) / (1-a))
    
    # ===== Multiclass/Categorial Cross Entropy =====
    @staticmethod
    def _mce_grad(a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a)
