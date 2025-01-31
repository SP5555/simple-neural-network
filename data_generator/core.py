import numpy as np
from .exceptions import InputValidationError

class DataGenerator:
    def __init__(self):
        self.rng = np.random.default_rng()

    def generate(self, n: int, name: str):
        generators = {
            'regression': self._generate_regression,
            'multilabel': self._generate_multilabel,
            'multiclass': self._generate_multiclass
        }
        name = name.strip().lower()
        if name in generators: return generators[name](n)
        raise InputValidationError(f"Unsupported generate function: {name}")
    
    def _generate_regression(self, n: int):
        # Make your own Data
        # Shape: (1, n)
        i1 = self.rng.uniform(-3, 3, size=n)
        i2 = self.rng.uniform(-3, 3, size=n)
        i3 = self.rng.uniform(-3, 3, size=n)
        i4 = self.rng.uniform(-3, 3, size=n)

        # Shape: (1, n)
        o1 = (i1*i4 + 5*i2 + 2*i1*i3 + i4)
        o2 = (4*i1 + 2*i2*i3 + 0.4*i4*i2 + 3*i3)
        o3 = (i1 + 0.3*i2 + 2*i3*i2 + 2*i4)

        # Shape: (n, count of input features)
        input_list = np.column_stack((i1, i2, i3, i4))

        # Shape: (n, count of output targets)
        output_list = np.column_stack((o1, o2, o3))

        # input noise
        input_list = self._add_noise(input_list, noise=0.5)

        return input_list.tolist(), output_list.tolist()

    def _generate_multilabel(self, n: int):
        # Make your own Data
        # Shape: (1, n)
        i1 = self.rng.uniform(-6, 6, size=n)
        i2 = self.rng.uniform(-6, 6, size=n)
        i3 = self.rng.uniform(-6, 6, size=n)
        i4 = self.rng.uniform(-6, 6, size=n)

        # Shape: (1, n)
        o1 = (i1*i4 - 5*i2 < 2*i1*i3 - i4).astype(float)
        o2 = (4*i1 - 2*i2*i3 + 0.4*i4*i2/i1 < -3*i3).astype(float)
        o3 = (-i1/i4 + 0.3*i2 - 8*i2*i2/i3 < 2*i4).astype(float)

        # Shape: (n, count of input features)
        input_list = np.column_stack((i1, i2, i3, i4))

        # Shape: (n, count of output targets)
        output_list = np.column_stack((o1, o2, o3))

        # input noise
        input_list = self._add_noise(input_list, noise=0.2)

        return input_list.tolist(), output_list.tolist()

    def _generate_multiclass(self, n: int):
        _input_features = 4
        _output_classes = 3

        # Make your own Data
        input_list = np.zeros((n, _input_features))
        output_list = np.zeros((n, _output_classes))

        # (1, n) shape array of random class labels
        class_labels = self.rng.integers(0, _output_classes, size=n)

        # Define input data ranges for each class
        class_data = {
            0: [(2, 5), (1, 5), (0, 4), (3, 5)],
            1: [(1, 4), (0, 3), (3, 6), (1, 5)],
            2: [(0, 3), (2, 6), (0, 5), (0, 2)],
        }

        for c in range(_output_classes):
            # extract indices of class c
            indices = np.where(class_labels == c)[0]

            # generate/fill up data in input list
            for i, (low, high) in enumerate(class_data[c]):
                input_list[indices, i] = self.rng.uniform(low, high, size=len(indices))
            # set correct class
            output_list[indices, c] = 1.0
        
        # input noise
        input_list = self._add_noise(input_list, noise=0.2)

        return input_list.tolist(), output_list.tolist()
    
    def _add_noise(self, data: np.ndarray, noise=0.5) -> np.ndarray:
        # Add uniform noise element-wise to the entire NumPy array
        return data + self.rng.uniform(-noise, noise, size=data.shape)
