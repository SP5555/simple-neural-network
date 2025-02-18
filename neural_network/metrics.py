import numpy as np
from .exceptions import InputValidationError
from .print_utils import PrintUtils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import NeuralNetwork

class Metrics:
    def __init__(self, core_instance: "NeuralNetwork"):
        self.core = core_instance
        self._check_batch_size = 1024

    def check_accuracy(self, test_input: list, test_output: list) -> None:
        if len(test_input) == 0 or len(test_output) == 0:
            raise InputValidationError("Datasets can't be empty.")
        if len(test_input) != len(test_output):
            raise InputValidationError("The sizes of input and output datasets must be equal.")
        if len(test_input[0]) != self.core._layers[0].input_size:
            raise InputValidationError("The input array size does not match the expected size for the neural network.")

        if self.core._layers[-1].activation.is_LL_regression_act:
            PrintUtils.print_info(f"Detected {self.core._layers[-1].activation.__class__.__name__} in the last layer. Running accuracy check for regression.")
            return self._regression_accuracy(test_input, test_output)
        if self.core._layers[-1].activation.is_LL_multilabel_act:
            PrintUtils.print_info(f"Detected {self.core._layers[-1].activation.__class__.__name__} in the last layer. Running accuracy check for multilabel.")
            return self._multilabel_accuracy(test_input, test_output)
        if self.core._layers[-1].activation.is_LL_multiclass_act:
            PrintUtils.print_info(f"Detected {self.core._layers[-1].activation.__class__.__name__} in the last layer. Running accuracy check for multiclass.")
            return self._multiclass_accuracy(test_input, test_output)
        PrintUtils.print_warning("The Accuracy Checker cannot determine the task type based on the current model configuration.\n" \
                                 "Ensure the last layer activation matches the intended task.")
    
    def _regression_accuracy(self, test_input: list, test_output: list) -> None:
        test_size = len(test_input)
        num_classes = len(test_output[0])

        # accumulated squared error per target
        ase = np.zeros(num_classes)

        index_start = 0
        while index_start < test_size:
            index_end = index_start + self._check_batch_size
            if index_end > test_size:
                index_end = test_size

            a: np.ndarray = np.array(test_input[index_start: index_end])
            o: np.ndarray = np.array(test_output[index_start: index_end])

            # forward pass
            predictions = self.core.forward_batch(a, raw_ndarray_output=True).T

            # accumulate squared error
            ase += np.sum(np.square(np.abs(predictions - o)), axis=0)

            index_start += self._check_batch_size
        
        mse = ase / test_size # mean squared error
        PrintUtils.print_info(f"Mean Squared Error on {test_size:,} samples")
        PrintUtils.print_info("Mean Squared Error per output: " + ''.join([f"{e:>8.2f}" for e in mse]))

    def _multilabel_accuracy(self, test_input: list, test_output: list) -> None:
        test_size = len(test_input)

        correct_predictions_count = 0
        index_start = 0
        while index_start < test_size:
            index_end = index_start + self._check_batch_size
            if index_end > test_size:
                index_end = test_size

            a: np.ndarray = np.array(test_input[index_start: index_end])
            o: np.ndarray = np.array(test_output[index_start: index_end])

            # forward pass
            predictions = self.core.forward_batch(a, raw_ndarray_output=True).T

            # 1 if checks, 0 if not.
            correct_predictions = np.abs(predictions - o) <= 0.5
            correct_predictions_count += np.sum(correct_predictions, axis=0)

            index_start += self._check_batch_size

        accuracy = correct_predictions_count / test_size * 100
        PrintUtils.print_info(f"Accuracy on {test_size:,} samples")
        PrintUtils.print_info("Accuracy per output: " + ''.join([f"{a:>8.2f}%" for a in accuracy]))

    def _multiclass_accuracy(self, test_input: list, test_output: list) -> None:
        test_size = len(test_input)
        num_classes = len(test_output[0])

        per_class_correct = np.zeros(num_classes)
        per_class_total = np.zeros(num_classes)

        correctly_categorized = 0
        index_start = 0
        while index_start < test_size:
            index_end = index_start + self._check_batch_size
            if index_end > test_size:
                index_end = test_size

            a: np.ndarray = np.array(test_input[index_start: index_end])
            o: np.ndarray = np.array(test_output[index_start: index_end])

            # forward pass
            predictions = self.core.forward_batch(a, raw_ndarray_output=True).T

            actual_class = np.argmax(o, axis=1)
            predicted_class = np.argmax(predictions, axis=1)
            correct_predictions = actual_class == predicted_class
            correctly_categorized += np.sum(correct_predictions, axis=0)

            for i in range(num_classes):
                per_class_correct[i] = np.sum((actual_class == i) & correct_predictions)
                per_class_total[i] = np.sum((actual_class == i))

            index_start += self._check_batch_size

        per_class_accuracy = per_class_correct / per_class_total * 100.0
        PrintUtils.print_info(f"Accuracy on {test_size:,} samples")
        PrintUtils.print_info("Accuracy per output: " + ''.join([f"{a:>8.2f}%" for a in per_class_accuracy]))
        cat_accuracy = correctly_categorized / test_size * 100.0
        PrintUtils.print_info(f"Overall categorization accuracy: {cat_accuracy:>8.2f}%")

    def compare_predictions(self, input: list, output: list) -> None:
        width_num = 6
        format_width = len(output[0]) * width_num
        print(f"{'Expected':>{format_width}} | {'Predicted':>{format_width}} | Input Data")

        predicted = self.core.forward_batch(input)
        for i in range(len(output)):
            print(''.join(f'{value:>{width_num}.2f}' for value in output[i]) + ' | ' +
                  ''.join(f'{value:>{width_num}.2f}' for value in predicted[i]) + ' | ' +
                  ''.join(f'{value:>6.2f}' for value in input[i]))
