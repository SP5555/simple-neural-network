import numpy as np
from .activations import Activations
from .print_utils import PrintUtils
from .exceptions import InputValidationError

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import NeuralNetwork

class Metrics:
    def __init__(self, core_instance: "NeuralNetwork"):
        self.core = core_instance

    def check_accuracy_classification(self, test_input: list, test_output: list) -> None:
        if self.core._act_func[-1].name not in Activations._LL_classification_acts:
            PrintUtils.print_warning("The Accuracy Classification function only works for models configured for classification tasks.")
            return
        if len(test_input) == 0 or len(test_output) == 0:
            raise InputValidationError("Datasets can't be empty.")
        if len(test_input) != len(test_output):
            raise InputValidationError("Input and Output data set sizes must be equal.")
        if len(test_input[0]) != self.core._layers[0]:
            raise InputValidationError("Input array size does not match the neural network.")
        
        _check_batch_size = 1024
        # threshold defines how close the prediction must be to expected to be considered correct
        # must be 0.0 < threshold <= 0.5
        _threshold = 0.5
        
        correct_predictions_count = 0

        test_size = len(test_input)
        index_start = 0
        correctly_categorized = 0
        while index_start < test_size:
            index_end = index_start + _check_batch_size
            if index_end > test_size:
                index_end = test_size

            a: np.ndarray = np.array(test_input[index_start: index_end])
            o: np.ndarray = np.array(test_output[index_start: index_end])

            # forward pass
            predictions = self.core.forward_batch(a, raw_ndarray_output=True).T

            if self.core._act_func[-1].name in Activations._LL_exclusive:
                actual_class = np.argmax(o, axis=1)
                predicted_class = np.argmax(predictions, axis=1)
                correct_predictions = actual_class == predicted_class
                correctly_categorized += np.sum(correct_predictions, axis=0)

            # 1 if checks, 0 if not.
            correct_predictions = np.abs(predictions - o) <= _threshold
            correct_predictions_count += np.sum(correct_predictions, axis=0)

            index_start += _check_batch_size

        accuracy = correct_predictions_count / test_size * 100
        PrintUtils.print_info(f"Accuracy on {test_size:,} samples")
        PrintUtils.print_info("Accuracy on each output: " + ''.join([f"{a:>8.2f}%" for a in accuracy]))
        if self.core._act_func[-1].name in Activations._LL_exclusive:
            cat_accuracy = correctly_categorized / test_size * 100
            PrintUtils.print_info(f"Overall categorization accuracy: {cat_accuracy:>8.2f}%")

    def compare_predictions(self, input: list, output: list) -> None:
        format_width = len(output[0]) * 9
        print(f"{'Expected':>{format_width}} | {'Predicted':>{format_width}} | Input Data")

        predicted = self.core.forward_batch(input)
        for i in range(len(output)):
            print(''.join(f'{value:>9.4f}' for value in output[i]) + ' | ' +
                  ''.join(f'{value:>9.4f}' for value in predicted[i]) + ' | ' +
                  ''.join(f'{value:>7.3f}' for value in input[i]))
