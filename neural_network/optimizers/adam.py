import numpy as np
from ..common import PrintUtils, ParamDict, InputValidationError
from .optimizer import Optimizer

class Adam(Optimizer):
    """
    Adaptive Moment Estimation
    =====

    Parameters
    ----------
    learn_rate : float
    
    beta1 : float

    beta2 : float

    Math
    ----
    m_(0) = 0.0
    v_(0) = 0.0

    m_(t) = beta1 * m_(t-1) + (1 - beta1) * grad(w_(t)) \\
    v_(t) = beta2 * v_(t-1) + (1 - beta2) * grad(w_(t))^2

    m_hat = m_(t) / (1 - beta1^t) \\
    v_hat = v_(t) / (1 - beta2^t)

    w_(t+1) = w_(t) - LR * m_hat / sqrt(v_hat + 1e-12)
    """
    def __init__(self, learn_rate: float, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__(learn_rate)

        # beta1: First moment decay (momentum decay)
        # beta2: Second moment decay (variance decay)
        # 0.0 disables the "gliding" behavior
        # 1.0 disables updates
        # MUST be less than 1.0
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0 :
            raise InputValidationError("Both beta1 and beta2 must be between within [0.0, 1.0).")
        if beta1 >= 0.95:
            PrintUtils.print_warning(f"Warning: beta1 = {beta1:.3f} may cause strong \"gliding\" behavior. " +
                                     "Consider keeping it less than 0.95")

        self.beta1 = beta1
        self.beta2 = beta2
        # 1st moment
        self.m = {}
        # 2nd moment
        self.v = {}
        # step counter
        self.t = 0

    def step(self, parameters: list[ParamDict]):
        self.t += 1

        for param in parameters:
            
            weight: np.ndarray = param['weight'].tensor

            param_id = id(param['weight'])
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(weight)
                self.v[param_id] = np.zeros_like(weight)
            
            # update 1st and 2nd moments
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * param['grad']
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * np.square(param['grad'])

            # this is some sort of scaling, known as "bias-correction"
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

            # update weights
            weight += -1 * self.LR * m_hat / np.sqrt(v_hat + 1e-12)

            param['weight'].assign(weight)

        self._clip_params(parameters)