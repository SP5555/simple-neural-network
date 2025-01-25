import numpy as np
from .activations import Activations
from .utils import Utils
from .metrics import Metrics
from .exceptions import InputValidationError

class NeuralNetwork:
    _LL_exclusive = (Activations.softmax,)
    
    def __init__(self,
                 layers: list,
                 activation: list[str] = [],
                 loss_function: str = "MSE",
                 learn_rate: float = 0.01,
                 lambda_parem: float = 0.0,
                 momentum: float = 0.8) -> None:
        
        self.utils = Utils(self)
        self.metrics = Metrics(self)
        
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
        if learn_rate >= 0.1:
            print(f"Warning: Learn rate {learn_rate:.3f} may cause instability. Consider keeping it less than 0.1.")
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
        # ReLU for regression
        # Sigmoid, Tanh for multilabel classification
        # Softmax for multiclass classification
        activation = [act.strip().lower() for act in activation] if activation else []

        if activation == []:
            # Default configuration
            activation = ["leaky_relu"] * (self._layer_count - 2) + ["sigmoid"]
        else:
            if len(activation) != self._layer_count - 1:
                raise InputValidationError(f"Expected {self._layer_count - 1} activation functions, but got {len(activation)}.")

            # Verifier
            for i, act in enumerate(activation):
                if i < self._layer_count - 2 and self.utils.get_activation_func(act) in self._LL_exclusive:
                    raise InputValidationError(f"{act} activation can't be used in hidden layers.")

        self._activation = [self.utils.get_activation_func(i) for i in activation]
        self._activation_derivative = [self.utils.get_activation_derivative_func(i) for i in activation]

        # Loss Functions
        # MSE for regression
        # BCE for multilabel classification
        # MCE/CCE for multiclass classification
        self._loss_derivative = self.utils.get_loss_derivative_func(loss_function)

        self.weights: list = []
        self.biases: list = []
        for i in range(self._layer_count - 1):
            # prevent extreme values
            self.weights.append(np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2/layers[i]))
            self.biases.append(np.random.randn(layers[i + 1], 1))
        
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

        print(f"Neural network with {layers} layers initialized.")
        print(f"Parameter Count: {self.utils.get_parameter_count():,}")

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
            a: np.ndarray = self._activation[i](z)

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
            a: np.ndarray = self._activation[i](z)

        if raw_ndarray_output:
            return a
        return a.T.tolist() # vanilla list, not np.ndarray

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

            # forward pass
            for i in range(self._layer_count - 1):

                # z = W*A + b
                z: np.ndarray = np.matmul(self.weights[i], a) + self.biases[i].reshape(-1, 1) # broadcasting
                # A = activation(z)
                a: np.ndarray = self._activation[i](z)
                # Record values for training
                z_layers.append(z)
                a_layers.append(a)

            # a holds columns of output here
            # y is desired output
            # derivative of loss function with respect to activations for LAST OUTPUT LAYER
            a_gradient_ith_layer: np.ndarray = self._loss_derivative(a, y)

            # important component for LAST LAYER backpropagation
            # term_2_3 = da(n)/dz(n) * dL/da(n)
            if self._activation[-1] == Activations.softmax:

                da_wrt_dz_reshaped = a_gradient_ith_layer[:, :, None].transpose(1, 0, 2) # (batch_size, dim, 1)
                dL_wrt_da_reshaped = Activations.softmax_derivative(z_layers[-1]) # Jacobians; (batch_size, dim, dim)
                
                t_2_3_3D = np.matmul(dL_wrt_da_reshaped, da_wrt_dz_reshaped) # (batch_size, dim, 1)
                term_2_3 = t_2_3_3D.squeeze(axis=-1).T # (dim, batch_size)
            else:
                term_2_3: np.ndarray = self._activation_derivative[-1](z_layers[-1]) * a_gradient_ith_layer
            
            # backpropagation
            for i in reversed(range(self._layer_count - 1)):

                # Math
                # z(n) = w(n)*a(n-1) + b(n)
                # a(n) = activation(z(n))

                # CALCULATE derivative of costs with respect to weights
                # dL/dw(n)
                # = dz(n)/dw(n) * da(n)/dz(n) * dL/da(n)
                # = a(n-1) * actv'(z(n)) * dL/da(n)
                w_gradient_ith_layer = np.matmul(term_2_3, a_layers[i].T) / batch_size

                # UPDATE/apply negative of average gradient change to weights
                lambda_w_old: np.ndarray = self.weights[i] * self.lambda_parem # Compute regularization term
                self.velocity_w[i] = self.momentum_beta * self.velocity_w[i] + (1 - self.momentum_beta) * (w_gradient_ith_layer + lambda_w_old)
                self.weights[i] += -1 * self.velocity_w[i] * self.learn_rate

                # CALCULATE derivative of costs with respect to biases
                # dL/db(n)
                # = dz(n)/db(n) * da(n)/dz(n) * dL/da(n)
                # = 1 * actv'(z(n)) * dL/da(n)
                b_gradient_ith_layer = np.sum(term_2_3, axis=1, keepdims=True) / batch_size

                # UPDATE/apply negative of average gradient change to biases / Update biases
                lambda_b_old: np.ndarray = self.biases[i] * self.lambda_parem # Compute regularization term
                self.velocity_b[i] = self.momentum_beta * self.velocity_b[i] + (1 - self.momentum_beta) * (b_gradient_ith_layer + lambda_b_old)
                self.biases[i] += -1 * self.velocity_b[i] * self.learn_rate

                if i == 0: continue # skip gradient descent calculation for input layer 
                # actual backpropagation
                # NOTE: a(n-1) affects all a(n), so backpropagation to a(n-1) will be related to all a(n)
                # dL/da(n-1)
                # = column-wise sum in w matrix [dz(n)/da(n-1) * da(n)/dz(n) * dL/da(n)]
                # = column-wise sum in w matrix [(w(n) * actv'(z(n)) * dL/da(n))]
                a_gradient_ith_layer = np.matmul(self.weights[i].T, term_2_3)
                
                # important component for HIDDEN LAYER backpropagations
                # update for next layer
                # term_2_3 = da(n)/dz(n) * dL/da(n)
                term_2_3: np.ndarray = self._activation_derivative[i-1](z_layers[i-1]) * a_gradient_ith_layer
            
            p: float = (100.0 * _ / epoch)
            print(f"Progress: {_+1:>5} / {epoch} [{p:>6.2f}%]  ", end='\r')
        
        print("===== ===== Training Completed ===== =====               ")

    def _get_parameter_count(self) -> int:
        c: int = 0
        for i in range(self._layer_count - 1):
            # c += self.layers[i + 1] * self.layers[i] # Weights
            # c += self.layers[i + 1] # Biases
            c += self.layers[i + 1] * (self.layers[i] + 1)
        return c
