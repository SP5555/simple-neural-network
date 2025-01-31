from ..exceptions import InputValidationError
from ..utils import Utils

class Layer:

    def __init__(self, input_size: int, output_size: int, activation: str):
        if input_size == 0:
            raise InputValidationError("A layer can't have 0 input.")
        if output_size == 0:
            raise InputValidationError("A layer can't have 0 output (0 neurons).")

        activation = activation.strip().lower()
        Utils._act_func_validator(activation)

        self.input_size = input_size
        self.output_size = output_size
        self.act_name = activation

    def build(self, is_first: bool, is_final: bool):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
    def optimize(self):
        raise NotImplementedError
    
    def _get_param_count(self):
        raise NotImplementedError
    
    @property
    def requires_training_flag(self):
        return False