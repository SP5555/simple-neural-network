import numpy as np

# LOSS FUNCTIONS
class Losses:
    # ===== Mean Squared Error =====
    @staticmethod
    def MSE_gradient(a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (a - y)
    
    # ===== Binary Cross Entropy =====
    @staticmethod
    def BCE_gradient(a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a) + ((1-y) / (1-a))
    
    # ===== Multiclass/Categorial Cross Entropy =====
    @staticmethod
    def MCE_gradient(a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a)
