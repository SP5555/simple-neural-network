import numpy as np
from .optimizer import Optimizer

class AdaGrad(Optimizer):
    """
    Adaptive Gradient

    G_(0) = 0.0

    G_(t) = G_(t-1) + grad(w_(t))^2
    LR_scaled = LR / sqrt(G_(t) + 1e-12)
    w_(t+1) = w_(t) - LR_scaled * grad(w_(t))
    """
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

        # accumulated squared gradients
        self.accu_sq_grad = {}
    
    def step(self, parameters: list[dict]) -> None:

        for param in parameters:

            param_id = id(param['weight'])
            if param_id not in self.accu_sq_grad:
                self.accu_sq_grad[param_id] = np.zeros_like(param['weight'])

            # accumulate squared gradients
            self.accu_sq_grad[param_id] += np.square(param['grad'])

            # calculate new learn rate
            LR_scaled = np.full_like(param['weight'], self.LR) / np.sqrt(self.accu_sq_grad[param_id] + 1e-12)

            # update weights
            param['weight'] += -1 * LR_scaled * param['grad']
