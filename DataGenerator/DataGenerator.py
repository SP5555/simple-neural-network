import random

class DataGenerator:
    def __init__(self):
        pass

    def generate(self, n: int, name: str):
        generators = {
            'multilabel': self._generate_multilabel,
            'multiclass': self._generate_multiclass
        }
        name = name.strip().lower()
        if name in generators: return generators[name](n)
        raise ValueError(f"Unsupported generate function: {name}")

    def _generate_multilabel(self, n: int):
        input_list: list = []
        output_list: list = []

        for _ in range(n):
            # Make your own Data
            i1: float = random.uniform(-6, 6)
            i2: float = random.uniform(-6, 6)
            i3: float = random.uniform(-6, 6)
            i4: float = random.uniform(-6, 6)
            o1: float = 0.0
            o2: float = 0.0
            o3: float = 0.0

            if i1*i1 - 5*i2 < 2*i1*i3 - i4:
                o1 = 1.0
            if 4*i1 - 2*i2*i3 + 0.4*i4*i2/i1 < -3*i3:
                o2 = 1.0
            if i1/i4 + 0.3*i2 - 8*i2*i2/i3 < 2*i4:
                o3 = 1.0

            # noise for inputs
            i1 += random.uniform(-0.2, 0.2)
            i2 += random.uniform(-0.2, 0.2)
            i3 += random.uniform(-0.2, 0.2)
            i4 += random.uniform(-0.2, 0.2)

            input_list.append([i1, i2, i3, i4])
            output_list.append([o1, o2, o3])
        return input_list, output_list

    def _generate_multiclass(self, n: int):
        input_list: list = []
        output_list: list = []

        for _ in range(n):
            # Make your own Data
            k = random.randint(0, 2)
            if (k == 0):
                o1 = 1.0
                o2 = 0.0
                o3 = 0.0
                i1: float = random.uniform(2, 5)
                i2: float = random.uniform(1, 5)
                i3: float = random.uniform(0, 4)
                i4: float = random.uniform(3, 5)
            elif (k == 1):
                o1 = 0.0
                o2 = 1.0
                o3 = 0.0
                i1: float = random.uniform(1, 4)
                i2: float = random.uniform(1, 3)
                i3: float = random.uniform(3, 6)
                i4: float = random.uniform(0, 5)
            else:
                o1 = 0.0
                o2 = 0.0
                o3 = 1.0
                i1: float = random.uniform(0, 3)
                i2: float = random.uniform(2, 6)
                i3: float = random.uniform(0, 6)
                i4: float = random.uniform(0, 2)

            input_list.append([i1, i2, i3, i4])
            output_list.append([o1, o2, o3])
        return input_list, output_list
    
class InputValidationError(Exception):
    def __str__(self):
        # Red color escape code for printing
        return f"\033[91m{self.args[0]}\033[0m" # Red text and reset after