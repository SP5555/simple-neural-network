from .optimizer import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent

    w_(t+1) = w_(t) - LR * grad(w_(t))
    """
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

    def step(self, parameters: list[dict]) -> None:

        for param in parameters:
            # update weights
            param['weight'] += -1 * self.LR * param['grad'] 

        self._clip_params(parameters)