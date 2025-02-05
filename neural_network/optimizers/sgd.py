from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learn_rate: float):
        super().__init__(learn_rate)

    def step(self, parameters: list[dict]) -> None:
        # parameter is a list of dictionaries
        # Keys: 'weight', 'grad'
        for param in parameters:
            # UPDATE weights
            param['weight'] += -1 * param['grad'] * self.LR
