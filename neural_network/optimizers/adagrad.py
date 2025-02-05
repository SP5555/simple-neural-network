import numpy as np
from .optimizer import Optimizer

class AdaGrad(Optimizer):
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

        # variable learn rates
        self.LRs = {}
        # accumulated squared gradients
        self.accu_sq_grad = {}
    
    def step(self, parameters: list[dict]) -> None:
        # parameter is a list of dictionaries
        # Keys: 'weight', 'grad'
        for param in parameters:

            param_id = id(param['weight'])
            if param_id not in self.LRs:
                self.LRs[param_id] = np.zeros_like(param['weight'])
                self.accu_sq_grad[param_id] = np.zeros_like(param['weight'])

            # accumulate squared gradients
            self.accu_sq_grad[param_id] += np.square(param['grad'])

            # UPDATE learn rate
            self.LRs[param_id] = np.full_like(param['weight'], self.LR) / np.sqrt(self.accu_sq_grad[param_id] + 1e-12)

            # UPDATE weights
            param['weight'] += -1 * param['grad'] * self.LRs[param_id]
