from ..auto_diff.auto_diff_reverse import Tensor, Square, Log, Abs, Clip
from ..exceptions import InputValidationError
import numpy as np

class Loss:
    """
    Abstract base class for all loss functions.

    This class serves as a template and cannot be used directly in the models. \\
    Methods in this class raise `NotImplementedError` to enforce implementation 
    in derived child classes.
    """
    def __init__(self):
        self.expression: Tensor = None

    def build_expression(self, A: Tensor, Y: Tensor):
        raise NotImplementedError
    
    def forward(self):
        self.expression.forward()

    def backward(self, seed: np.ndarray):
        self.expression.backward(seed)

# ===== Mean Absolute Error =====
class MAE(Loss):
    def build_expression(self, A: Tensor, Y: Tensor):
        self.expression = Abs(A - Y)

# ===== Mean Squared Error =====
class MSE(Loss):
    def build_expression(self, A: Tensor, Y: Tensor):
        self.expression = Square(A - Y)

# ===== Huber Loss =====
class Huber(Loss):
    def __init__(self, delta: float):
        if delta <= 0.0:
            raise InputValidationError("Huber constant delta must be positive.")
        self.d = delta

    def build_expression(self, A: Tensor, Y: Tensor):
        diff = A - Y
        diff.forward()
        mid = Tensor(np.abs(diff.tensor) <= self.d)
        pos = Tensor(diff.tensor > self.d)
        neg = Tensor(diff.tensor < -self.d)

        mid_expression = (Tensor(0.5) * Square(diff) * mid)
        p_expression = (Tensor(self.d) * (diff - Tensor(self.d/2)) * pos)
        n_expression = (Tensor(self.d) * (-diff - Tensor(self.d/2)) * neg)

        self.expression = mid_expression + p_expression + n_expression

# ===== Binary Cross Entropy =====
class BCE(Loss):
    def build_expression(self, A: Tensor, Y: Tensor):
        bound = 1e-12
        A_c = Clip(A, bound, 1-bound)
        
        self.expression = -(Y * Log(A_c) + (Tensor(1.0) - Y) * Log(Tensor(1.0) - A_c))

# ===== Multiclass/Categorial Cross Entropy =====
class CCE(Loss):
    def build_expression(self, A: Tensor, Y: Tensor):
        bound = 1e-12
        A_c = Clip(A, bound, 1-bound)

        self.expression = -(Y * Log(A_c))
