import numpy as np
from .exceptions import InputValidationError
from .print_utils import PrintUtils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core import NeuralNetwork

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

        last_layer = next((l for l in reversed(self.core._layers) if hasattr(l, '_activation')), None)

        if last_layer is None:
            PrintUtils.print_warning("The Accuracy Checker cannot find a layer with an activation function.")
            return

        if last_layer._activation.is_LL_regression_act:
            PrintUtils.print_info(f"Detected {last_layer._activation.__class__.__name__} in the last layer. Running accuracy check for regression.")
            return self._regression_accuracy(test_input, test_output)
        if last_layer._activation.is_LL_multilabel_act:
            PrintUtils.print_info(f"Detected {last_layer._activation.__class__.__name__} in the last layer. Running accuracy check for multilabel.")
            return self._multilabel_accuracy(test_input, test_output)
        if last_layer._activation.is_LL_multiclass_act:
            PrintUtils.print_info(f"Detected {last_layer._activation.__class__.__name__} in the last layer. Running accuracy check for multiclass.")
            return self._multiclass_accuracy(test_input, test_output)
        PrintUtils.print_warning("The Accuracy Checker cannot determine the task type based on the current model configuration.\n"
                                 "Ensure the last layer activation matches the intended task.")

    def _regression_accuracy(self, test_input: list, test_output: list) -> None:
        test_size = len(test_input)
        n_outputs = len(test_output[0])

        sse    = np.zeros(n_outputs)  # sum of squared errors  -> RMSE
        sae    = np.zeros(n_outputs)  # sum of absolute errors -> MAE
        sum_y  = np.zeros(n_outputs)  # sum of true values     \
        sum_y2 = np.zeros(n_outputs)  # sum of true values^2   -> R2

        i = 0
        while i < test_size:
            a    = np.array(test_input[i  : i + self._check_batch_size])
            o    = np.array(test_output[i : i + self._check_batch_size])
            pred = self.core.forward_batch(a, raw_ndarray_output=True).T
            sse    += np.sum(np.square(pred - o), axis=0)
            sae    += np.sum(np.abs(pred - o),    axis=0)
            sum_y  += np.sum(o,            axis=0)
            sum_y2 += np.sum(np.square(o), axis=0)
            i += self._check_batch_size

        rmse   = np.sqrt(sse / test_size)
        mae    = sae / test_size
        ss_tot = sum_y2 - (sum_y ** 2) / test_size
        r2     = np.where(ss_tot > 0, 1.0 - sse / ss_tot, 0.0)

        w = 10
        PrintUtils.print_info(f"Regression Metrics on {test_size:,} samples")
        PrintUtils.print_info(f"  {'':10}{'RMSE':>{w}}{'MAE':>{w}}{'R2':>{w}}")
        for i in range(n_outputs):
            PrintUtils.print_info(f"  Output {i+1:<4}{rmse[i]:>{w}.4f}{mae[i]:>{w}.4f}{r2[i]:>{w}.4f}")

    def _multilabel_accuracy(self, test_input: list, test_output: list) -> None:
        test_size = len(test_input)
        n_labels  = len(test_output[0])

        tp          = np.zeros(n_labels)
        fp          = np.zeros(n_labels)
        fn          = np.zeros(n_labels)
        exact_match = 0

        i = 0
        while i < test_size:
            a      = np.array(test_input[i  : i + self._check_batch_size])
            o      = np.array(test_output[i : i + self._check_batch_size])
            pred   = self.core.forward_batch(a, raw_ndarray_output=True).T
            binary = (pred >= 0.5).astype(float)
            tp += np.sum((binary == 1) & (o == 1), axis=0)
            fp += np.sum((binary == 1) & (o == 0), axis=0)
            fn += np.sum((binary == 0) & (o == 1), axis=0)
            exact_match += int(np.sum(np.all(binary == o, axis=1)))
            i += self._check_batch_size

        precision  = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall     = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1         = np.where(precision + recall > 0,
                              2 * precision * recall / (precision + recall), 0.0)
        exact_rate = exact_match / test_size * 100.0

        w = 10
        PrintUtils.print_info(f"Multilabel Metrics on {test_size:,} samples  (threshold = 0.5)")
        PrintUtils.print_info(f"  {'':10}{'Precision':>{w}}{'Recall':>{w}}{'F1':>{w}}")
        for i in range(n_labels):
            PrintUtils.print_info(f"  Label  {i+1:<4}{precision[i]:>{w}.4f}{recall[i]:>{w}.4f}{f1[i]:>{w}.4f}")
        PrintUtils.print_info(f"  Exact Match (all labels correct): {exact_rate:.2f}%")

    def _multiclass_accuracy(self, test_input: list, test_output: list) -> None:
        test_size = len(test_input)
        n_classes = len(test_output[0])

        tp              = np.zeros(n_classes)
        fp              = np.zeros(n_classes)
        fn              = np.zeros(n_classes)
        overall_correct = 0

        i = 0
        while i < test_size:
            a         = np.array(test_input[i  : i + self._check_batch_size])
            o         = np.array(test_output[i : i + self._check_batch_size])
            pred      = self.core.forward_batch(a, raw_ndarray_output=True).T
            actual    = np.argmax(o,    axis=1)
            predicted = np.argmax(pred, axis=1)
            overall_correct += int(np.sum(actual == predicted))
            for c in range(n_classes):
                tp[c] += np.sum((predicted == c) & (actual == c))
                fp[c] += np.sum((predicted == c) & (actual != c))
                fn[c] += np.sum((predicted != c) & (actual == c))
            i += self._check_batch_size

        precision   = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall      = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1          = np.where(precision + recall > 0,
                               2 * precision * recall / (precision + recall), 0.0)
        overall_acc = overall_correct / test_size * 100.0

        w = 10
        PrintUtils.print_info(f"Multiclass Metrics on {test_size:,} samples")
        PrintUtils.print_info(f"  {'':10}{'Precision':>{w}}{'Recall':>{w}}{'F1':>{w}}")
        for i in range(n_classes):
            PrintUtils.print_info(f"  Class  {i+1:<4}{precision[i]:>{w}.4f}{recall[i]:>{w}.4f}{f1[i]:>{w}.4f}")
        PrintUtils.print_info(f"  Overall Accuracy: {overall_acc:.2f}%")

    def compare_predictions(self, input: list, output: list) -> None:
        width_num    = 6
        format_width = len(output[0]) * width_num
        print(f"{'Expected':>{format_width}} | {'Predicted':>{format_width}} | Input Data")

        predicted = self.core.forward_batch(input)
        for i in range(len(output)):
            print(''.join(f'{value:>{width_num}.2f}' for value in output[i]) + ' | ' +
                  ''.join(f'{value:>{width_num}.2f}' for value in predicted[i]) + ' | ' +
                  ''.join(f'{value:>6.2f}' for value in input[i]))
