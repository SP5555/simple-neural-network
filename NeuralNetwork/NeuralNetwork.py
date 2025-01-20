import numpy as np

# Less than 500 neurons in each layer suggested, or be evil
class NeuralNetwork:
    def __init__(self,
                 layers: list,
                 activation_hidden: str = 'leaky_relu',
                 activation_output: str = 'leaky_relu',
                 loss_function: str = 'MSE',
                 learn_rate: float = 0.01,
                 lambda_parem: float = 0.0,
                 momentum: float = 0.8) -> None:
        
        if not layers: # if list is empty
            raise InputValidationError("Empty layer configuration not possible.")
        if 0 in layers: # if layer with 0 neurons exists
            raise InputValidationError("A layer can't have 0 neurons.")
        if not all(isinstance(x, int) for x in layers): # if not all entries are integers
            raise InputValidationError("Layers must be a list of integers.")
        
        self.layers: list = layers
        self._layer_count: int = len(layers)
        
        # Learn Rate
        # How fast or slow this network learns
        #     new_parameter = old_parameter - velocity * learn_rate 
        if learn_rate <= 0.0:
            raise InputValidationError("Learn rate must be positive.")
        if learn_rate >= 1.0:
            print(f"Warning: Learn rate {learn_rate:.3f} may cause instability. Consider keeping it less than 1.0.")
        self.learn_rate = learn_rate

        # Regularization Strength
        # Large weights and biases will change more aggressively than small ones
        # Don't set it too large, at most 0.01 (unless you know what you're doing)
        #     new_velocity = momentum * old_velocity + (1-momentum) * (parameter_gradient + lambda_parem * parameter)
        if lambda_parem < 0.0:
            raise InputValidationError("Regularization Strength can't be negative.")
        if lambda_parem > 0.01:
            print(f"Warning: Regularization Strength {lambda_parem:.3f} is strong. Consider keeping it less than 0.01")
        self.lambda_parem = lambda_parem

        # Momentum Beta for Momentum Gradient Descent
        # 0.0 disables the momentum behavior
        # having momentum beta helps escape the high cost "plateaus" better
        # high values result in smoother/stronger "gliding" descent
        # MUST be less than 1.0
        #     new_velocity = momentum * old_velocity + (1-momentum) * (parameter_gradient + lambda_parem * parameter)
        if momentum < 0.0:
            raise InputValidationError("Momentum can't be negative.")
        if momentum >= 1.0:
            raise InputValidationError("Momentum must be less than 1.0.")
        if momentum >= 0.95:
            print(f"Warning: Momentum value {momentum:.3f} may cause strong \"gliding\" behavior. Consider keeping it less than 0.95")
        self.momentum_beta = momentum

        # Activation Functions
        # Sigmoid, Tanh for probability distribution, multilabel classification
        # ReLU for regression
        # Softmax for multiclass classification
        self._activation = self._get_activation_func(activation_hidden)
        self._activation_derivative = self._get_activation_derivative_func(activation_hidden)

        self._activation_last_layer = self._get_activation_func(activation_output)
        self._activation_last_layer_derivative = self._get_activation_derivative_func(activation_output)

        # Loss Functions
        # MSE for regression
        # BCE for binary classification
        # MCE for multiclass classification (not available yet)
        self._loss_derivative = self._get_loss_derivative_func(loss_function)

        self.weights: list = []
        self.biases: list = []
        for i in range(self._layer_count - 1):
            # prevent extreme values
            self.weights.append(np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2/layers[i]))
            self.biases.append(np.random.randn(layers[i + 1], 1))
        
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

        print(f"Neural network with {layers} layers initialized.")
        print(f"Parameter Count: {self._get_parameter_count():,}")
    
    # ACTIVATION FUNCTIONS
    # ===== Sigmoid =====
    def _sigmoid(self, np_array: np.ndarray) -> np.ndarray:
        # s(x) = (tanh(x/2) + 1) / 2
        return (np.tanh(np_array / 2) + 1) / 2
    
    def _sigmoid_derivative(self, np_array: np.ndarray) -> np.ndarray:
        # s'(x) = s(x) * (1 - s(x))
        s: np.ndarray = (np.tanh(np_array / 2) + 1) / 2
        return s*(1-s)
    
    # ===== Tanh =====
    def _tanh(self, np_array: np.ndarray) -> np.ndarray:
        return np.tanh(np_array)

    def _tanh_derivative(self, np_array: np.ndarray) -> np.ndarray:
        # tanh'(x) = 1 - tanh(x)^2
        t = np.tanh(np_array)
        return 1 - t*t

    # ===== ReLU =====
    def _relu(self, np_array: np.ndarray) -> np.ndarray:
        return np.maximum(0, np_array)
        
    def _relu_derivative(self, np_array: np.ndarray) -> np.ndarray:
        return np.where(np_array > 0, 1, 0)

    # ===== Leaky ReLU =====
    def _leaky_relu(self, np_array: np.ndarray) -> np.ndarray:
        return np.where(np_array > 0, np_array, 0.1 * np_array)
        
    def _leaky_relu_derivative(self, np_array: np.ndarray) -> np.ndarray:
        return np.where(np_array > 0, 1, 0.1)
    
    # ===== Softmax =====
    def _softmax(self, np_array: np.ndarray) -> np.ndarray:
        exp = np.exp(np_array - np.max(np_array, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    def _softmax_derivative(self, np_array: np.ndarray) -> np.ndarray:
        dim = np_array.shape[0]
        softmax = self._softmax(np_array) # Shape: (dim, batch_size)

        softmax_expanded = softmax.T[:, :, None]  # Shape: (batch_size, dim, 1)

        # Shape: (batch_size, dim, dim)
        # matmul perform matrix multiplication on the last two dimensions here
        # each sample "slice" on 0-th axis is: I * M_softmax(dim, 1) - np.dot(M_softmax, M_softmax.T)
        jacobians = np.eye(dim)[None, :, :] * softmax_expanded - np.matmul(softmax_expanded, softmax_expanded.transpose(0, 2, 1))

        return jacobians.transpose(1, 2, 0) # Shape: (dim, dim, batch_size)

    # LOSS FUNCTIONS
    # ===== Mean Squared Error =====
    def _MSE_gradient(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (a - y)
    
    # ===== Binary Cross Entropy =====
    def _BCE_gradient(self, a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a) + ((1-y) / (1-a))
    
    # ===== Multiclass/Categorial Cross Entropy =====
    def _MCE_gradient(self, a: np.ndarray, y:np.ndarray) -> np.ndarray:
        bound = 1e-12
        a = np.clip(a, bound, 1-bound)
        return -(y/a)

    def _get_activation_func(self, name: str):
        actv_funcs = {
            'relu': self._relu,
            'leaky_relu': self._leaky_relu,
            'tanh': self._tanh,
            'sigmoid': self._sigmoid,
            'softmax': self._softmax
        }
        name = name.strip().lower()
        if name in actv_funcs: return actv_funcs[name]
        raise ValueError(f"Unsupported activation function: {name}")

    def _get_activation_derivative_func(self, name: str):
        actv_deriv_funcs = {
            'relu': self._relu_derivative,
            'leaky_relu': self._leaky_relu_derivative,
            'tanh': self._tanh_derivative,
            'sigmoid': self._sigmoid_derivative,
            'softmax': self._softmax_derivative
        }
        name = name.strip().lower()
        if name in actv_deriv_funcs: return actv_deriv_funcs[name]
        raise ValueError(f"Unsupported activation function: {name}")

    def _get_loss_derivative_func(self, name: str):
        loss_funcs = {
            'mse': self._MSE_gradient,
            'bce': self._BCE_gradient,
            'mce': self._MCE_gradient,
            'cce': self._MCE_gradient # same as mce
        }
        name = name.strip().lower()
        if name in loss_funcs: return loss_funcs[name]
        raise ValueError(f"Unsupported loss function: {name}")

    def _get_parameter_count(self) -> int:
        c: int = 0
        for i in range(self._layer_count - 1):
            # c += self.layers[i + 1] * self.layers[i] # Weights
            # c += self.layers[i + 1] # Biases
            c += self.layers[i + 1] * (self.layers[i] + 1)
        return c
    
    def inspect_weights_and_biases(self) -> None:
        np.set_printoptions(precision=4)
        for i in range(self._layer_count - 1):
            print(f'w L{i+1} -> L{i+2}')
            print(self.weights[i])
            print(f'b L{i+1} -> L{i+2}')
            print(self.biases[i])
    
    # main feed forward function (single)
    def forward(self, input: list) -> list:
        if len(input) != self.layers[0]:
            raise InputValidationError("Input array size does not match the neural network.")

        # activation
        # changes 1D input array into n x 1 sized numpy 2D array
        a: np.ndarray = np.array(input).reshape(-1, 1)

        # forward pass
        for i in range(self._layer_count - 1):
            # z = W*A + b
            z: np.ndarray = np.matmul(self.weights[i], a) + self.biases[i]
            # A = activation(z)
            if i < self._layer_count - 2:
                a: np.ndarray = self._activation(z)
            else:
                a: np.ndarray = self._activation_last_layer(z)
        
        return a.flatten().tolist()

    # main feed forward function (multiple)
    def forward_batch(self, input: list, raw_ndarray_output = False) -> np.ndarray:
        if len(input) == 0:
            raise InputValidationError("Input batch does not have data.")
        if len(input[0]) != self.layers[0]:
            raise InputValidationError("Input array size does not match the neural network.")
        
        # activation
        # changes an array of inputs into n x batch_size numpy 2D array
        a: np.ndarray = np.array(input).T

        # forward pass
        for i in range(self._layer_count - 1):
            # z = W*A + b
            z: np.ndarray = np.matmul(self.weights[i], a) + self.biases[i].reshape(-1, 1) # broadcasting
            # A = activation(z)
            if i < self._layer_count - 2:
                a: np.ndarray = self._activation(z)
            else:
                a: np.ndarray = self._activation_last_layer(z)
        
        if raw_ndarray_output:
            return a
        return a.T.tolist() # user friendly format, not np.ndarray
    
    def train(self,
              input_list: list,
              output_list: list,
              epoch: int = 100,
              batch_size: int = 32) -> None:
        if len(input_list) == 0 or len(output_list) == 0:
            raise InputValidationError("Datasets can't be empty.")
        if len(input_list) != len(output_list):
            raise InputValidationError("Input and Output data set sizes must be equal.")
        if len(input_list[0]) != self.layers[0]:
            raise InputValidationError("Input array size does not match the neural network.")
        if len(output_list[0]) != self.layers[-1]:
            raise InputValidationError("Output array size does not match the neural network.")
        if epoch <= 0:
            raise InputValidationError("Epoch must be positive.")
        
        if batch_size > len(input_list): batch_size = len(input_list)

        input_ndarray = np.array(input_list)
        output_ndarray = np.array(output_list)
        
        for _ in range(epoch):
            # pick random candidates as train data in each epoch
            indices = np.random.choice(len(input_list), size=batch_size, replace=False)
            i_batch = input_ndarray[indices]
            o_batch = output_ndarray[indices]

            # activation
            # a contains columns of sample inputs, each column is an individual sample
            a: np.ndarray = i_batch.T
            
            # desired output
            # same as a
            y: np.ndarray = o_batch.T

            # These are required for training
            # activation of hidden and final layers
            a_layers: list = []
            a_layers.append(a) # input layer is appended
            # W*A + b of hidden and final layers
            z_layers: list = []

            # individual gradients
            # (C gradient with respect to w,a,b)
            w_gradient_layers: list = [] # required, core
            b_gradient_layers: list = [] # required, core

            # forward pass
            for i in range(self._layer_count - 1):

                # z = W*A + b
                z: np.ndarray = np.matmul(self.weights[i], a) + self.biases[i].reshape(-1, 1) # broadcasting
                # A = activation(z)
                if i < self._layer_count - 2:
                    a: np.ndarray = self._activation(z)
                else:
                    a: np.ndarray = self._activation_last_layer(z)
                # Record values for training
                z_layers.append(z)
                a_layers.append(a)
  
            # a holds columns of output here
            # y is desired output
            # derivative of loss function with respect to activations for LAST OUTPUT LAYER
            a_gradient_idv_layer: np.ndarray = self._loss_derivative(a, y)

            # important component for LAST LAYER backpropagation
            # term_2_3 = da(n)/dz(n) * dL/da(n)
            if self._activation_last_layer == self._softmax:
                softmax_derivative = self._softmax_derivative(z_layers[-1]) # (dim, dim, batch_size)

                da_wrt_dz_reshaped = a_gradient_idv_layer[:, :, None].transpose(1, 0, 2) # (batch_size, dim, 1)
                dL_wrt_dz_reshaped = softmax_derivative.transpose(2, 0, 1) # (batch_size, dim, dim)
                
                t_2_3_3D = np.matmul(dL_wrt_dz_reshaped, da_wrt_dz_reshaped) # (batch_size, dim, 1)
                term_2_3 = t_2_3_3D.squeeze(axis=-1).T # (dim, batch_size)
            else:
                term_2_3: np.ndarray = self._activation_last_layer_derivative(z_layers[-1]) * a_gradient_idv_layer
            
            # backpropagation
            for i in reversed(range(self._layer_count - 1)):

                # Math
                # z(n) = w(n)*a(n-1) + b(n)
                # a(n) = activation(z(n))

                # important component for HIDDEN LAYER backpropagations
                # term_2_3 = da(n)/dz(n) * dL/da(n)
                if i < self._layer_count - 2:
                    term_2_3: np.ndarray = self._activation_derivative(z_layers[i]) * a_gradient_idv_layer

                # derivative of costs with respect to weights
                # dL/dw(n)
                # = dz(n)/dw(n) * da(n)/dz(n) * dL/da(n)
                # = a(n-1) * actv'(z(n)) * dL/da(n)
                w_gradient_idv_layer = np.matmul(term_2_3, a_layers[i].T)
                w_gradient_layers.insert(0, w_gradient_idv_layer / batch_size)

                # derivative of costs with respect to biases
                # dL/db(n)
                # = dz(n)/db(n) * da(n)/dz(n) * dL/da(n)
                # = 1 * actv'(z(n)) * dL/da(n)
                b_gradient_aggregated = np.sum(term_2_3, axis=1, keepdims=True)
                b_gradient_layers.insert(0, b_gradient_aggregated / batch_size)

                if i == 0: continue # skip gradient descent calculation for input layer 
                # actual backpropagation
                # NOTE: a(n-1) affects all a(n), so backpropagation to a(n-1) will be related to all a(n)
                # dL/da(n-1)
                # = column-wise sum in w matrix [dz(n)/da(n-1) * da(n)/dz(n) * dL/da(n)]
                # = column-wise sum in w matrix [(w(n) * actv'(z(n)) * dL/da(n))]
                new_a_gradient_idv_layer = np.matmul(self.weights[i].T, term_2_3)
                a_gradient_idv_layer = new_a_gradient_idv_layer

            # apply negative of average gradient change to weights and biases
            for i in range(self._layer_count - 1):
                # Update weights
                # Compute regularization term
                lambda_w_old: np.ndarray = self.weights[i] * self.lambda_parem
                self.velocity_w[i] = self.momentum_beta * self.velocity_w[i] + (1 - self.momentum_beta) * (w_gradient_layers[i] + lambda_w_old)
                self.weights[i] += -1 * self.velocity_w[i] * self.learn_rate

                # Update biases
                # Compute regularization term
                lambda_b_old: np.ndarray = self.biases[i] * self.lambda_parem
                self.velocity_b[i] = self.momentum_beta * self.velocity_b[i] + (1 - self.momentum_beta) * (b_gradient_layers[i] + lambda_b_old)
                self.biases[i] += -1 * self.velocity_b[i] * self.learn_rate
            
            p: float = (100.0 * _ / epoch)
            print(f"Progress: {_+1:>5} / {epoch} [{p:>6.2f}%]  ", end='\r')
        
        print("===== ===== Training Completed ===== =====               ")
    
    def check_accuracy_classification(self, test_input: list, test_output: list) -> None:
        _check_batch_size = 1024
        # threshold defines how close the prediction must be to expected to be considered correct
        # must be 0.0 < threshold <= 0.5
        _threshold = 0.5

        if self._activation_last_layer not in (self._sigmoid, self._tanh, self._softmax):
           raise InputValidationError("Model is not set up for classification.")
        if len(test_input) == 0 or len(test_output) == 0:
            raise InputValidationError("Datasets can't be empty.")
        if len(test_input) != len(test_output):
            raise InputValidationError("Input and Output data set sizes must be equal.")
        if len(test_input[0]) != self.layers[0]:
            raise InputValidationError("Input array size does not match the neural network.")
        
        correct_predictions_count = 0

        test_size = len(test_input)
        index_start = 0
        while index_start < test_size:
            index_end = index_start + _check_batch_size
            if index_end > test_size:
                index_end = test_size

            a: np.ndarray = np.array(test_input[index_start: index_end])

            # forward pass
            a = self.forward_batch(a, raw_ndarray_output=True).T

            # a holds rows of predicted output here
            # shape is batch_size x n

            # o is expected output
            o: np.ndarray = np.array(test_output[index_start: index_end])

            # 1 if checks, 0 if not.
            correct_predictions = np.logical_or(
                np.logical_and(a >= (1.0 - _threshold), o == 1.0),
                np.logical_and(a < _threshold, o == 0.0),
            )
            correct_predictions_count += np.sum(correct_predictions, axis=0)

            index_start += _check_batch_size
        accuracy = correct_predictions_count / test_size * 100
        print(f"Accuracy on {test_size} samples: " + ''.join([f"{a:>8.2f}%" for a in accuracy]))
    
    def compare_predictions(self, input: list, output: list) -> None:
        format_width = len(output[0]) * 8
        print(f"{"Expected":>{format_width}} | {"Predicted":>{format_width}} | Input Data")

        predicted = self.forward_batch(input)
        for i in range(len(output)):
            print(''.join(f'{value:>8.4f}' for value in output[i]) + ' | ' +
                  ''.join(f'{value:>8.4f}' for value in predicted[i]) + ' | ' +
                  ''.join(f'{value:>7.3f}' for value in input[i]))

class InputValidationError(Exception):
    def __str__(self):
        # Red color escape code for printing
        return f"\033[91m{self.args[0]}\033[0m" # Red text and reset after