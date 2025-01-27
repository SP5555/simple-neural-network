import numpy as np
from .activations import Activations
from .utils import Utils
from .metrics import Metrics
from .exceptions import InputValidationError

class NeuralNetwork:
    
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
        if len(layers) == 1:
            raise InputValidationError("Must have at least 2 layers (Input and Output).")
        if 0 in layers: # if layer with 0 neurons exists
            raise InputValidationError("A layer can't have 0 neurons.")
        if not all(isinstance(x, int) for x in layers): # if not all entries are integers
            raise InputValidationError("Layers must be a list of integers.")
        
        self._layers: list = layers
        self._layer_count: int = len(layers)
        
        # Learn Rate
        # How fast or slow this network learns
        #     new_parameter = old_parameter - velocity * learn_rate 
        if learn_rate <= 0.0:
            raise InputValidationError("Learn rate must be positive.")
        if learn_rate >= 0.1:
            print(f"Warning: Learn rate {learn_rate:.3f} may cause instability. Consider keeping it less than 0.1.")
        self.lr = learn_rate

        # L2 Regularization Strength
        # low reg strength -> cook in class, and fail in exam; overfit
        # high reg strength -> I am dumb dumb, can't learn; underfit
        # Large weights and biases will are penalized more aggressively than small ones
        # Don't set it too large, at most 0.01 (unless you know what you're doing)
        #     new_velocity = momentum * old_velocity + (1-momentum) * (parameter_gradient + lambda_parem * parameter)
        if lambda_parem < 0.0:
            raise InputValidationError("Regularization Strength can't be negative.")
        if lambda_parem > 0.01:
            print(f"Warning: Regularization Strength {lambda_parem:.3f} is strong. Consider keeping it less than 0.01")
        self.l2_lambda = lambda_parem

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
        self.m_beta = momentum

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
                if i < self._layer_count - 2 and self.utils._get_act_func(act) in Activations._LL_exclusive:
                    raise InputValidationError(f"{act} activation can't be used in hidden layers.")

        self._act_func = [self.utils._get_act_func(i) for i in activation]
        self._act_deriv_func = [self.utils._get_act_deriv_func(i) for i in activation]

        # Loss Functions
        # MSE for regression
        # BCE for multilabel classification
        # MCE/CCE for multiclass classification
        self._loss_deriv = self.utils._get_loss_deriv_func(loss_function)

        self.weights: list = []
        self.biases: list = []
        for i in range(self._layer_count - 1):
            # prevent extreme values
            self.weights.append(np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2/layers[i]))
            self.biases.append(np.random.randn(layers[i + 1], 1))
        
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]

        print(f"Neural network with {layers} layers initialized.")
        print(f"Parameter Count: {self.utils._get_param_count():,}")

         # main feed forward function (single)
    def forward(self, input: list) -> list:
        if len(input) != self._layers[0]:
            raise InputValidationError("Input array size does not match the neural network.")

        # activation
        # changes 1D input array into n x 1 sized numpy 2D array
        a: np.ndarray = np.array(input).T

        # forward pass
        for i in range(self._layer_count - 1):
            # z = W*A + b
            z: np.ndarray = np.matmul(self.weights[i], a) + self.biases[i]
            # A = activation(z)
            a: np.ndarray = self._act_func[i](z)

        return a.flatten().tolist()

    # main feed forward function (multiple)
    def forward_batch(self, input: list, raw_ndarray_output = False) -> np.ndarray:
        if len(input) == 0:
            raise InputValidationError("Input batch does not have data.")
        if len(input[0]) != self._layers[0]:
            raise InputValidationError("Input array size does not match the neural network.")
        
        # activation
        # changes an array of inputs into n x batch_size numpy 2D array
        a: np.ndarray = np.array(input).T

        # forward pass
        for i in range(self._layer_count - 1):
            # z = W*A + b
            z: np.ndarray = np.matmul(self.weights[i], a) + self.biases[i].reshape(-1, 1) # broadcasting
            # A = activation(z)
            a: np.ndarray = self._act_func[i](z)

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
        if len(input_list[0]) != self._layers[0]:
            raise InputValidationError("Input array size does not match the neural network.")
        if len(output_list[0]) != self._layers[-1]:
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
                a: np.ndarray = self._act_func[i](z)
                # Record values for training
                z_layers.append(z)
                a_layers.append(a)

            # a holds columns of output here
            # y is desired output
            # derivative of loss function with respect to activations for LAST OUTPUT LAYER
            act_grad: np.ndarray = self._loss_deriv(a, y)

            # important component for LAST LAYER backpropagation
            # term_2_3 = da(n)/dz(n) * dL/da(n)
            if self._act_func[-1] == Activations._softmax:

                da_wrt_dz = act_grad[:, :, None].transpose(1, 0, 2) # (batch_size, dim, 1)
                dL_wrt_da = Activations._softmax_deriv(z_layers[-1]) # Jacobians; (batch_size, dim, dim)
                
                t_2_3_3D = np.matmul(dL_wrt_da, da_wrt_dz) # (batch_size, dim, 1)
                term_2_3 = t_2_3_3D.squeeze(axis=-1).T # (dim, batch_size)
            else:
                term_2_3: np.ndarray = self._act_deriv_func[-1](z_layers[-1]) * act_grad
            
            # backpropagation
            for i in reversed(range(self._layer_count - 1)):

                # Math
                # z(n) = w(n)*a(n-1) + b(n)
                # a(n) = activation(z(n))

                # CALCULATE derivative of costs with respect to weights
                # dL/dw(n)
                # = dz(n)/dw(n) * da(n)/dz(n) * dL/da(n)
                # = a(n-1) * actv'(z(n)) * dL/da(n)
                w_grad = np.matmul(term_2_3, a_layers[i].T) / batch_size

                # UPDATE/apply negative of average gradient change to weights
                l2_term_for_w: np.ndarray = self.weights[i] * self.l2_lambda # Compute regularization term
                self.v_w[i] = self.m_beta * self.v_w[i] + (1 - self.m_beta) * (w_grad + l2_term_for_w)
                self.weights[i] += -1 * self.v_w[i] * self.lr

                # CALCULATE derivative of costs with respect to biases
                # dL/db(n)
                # = dz(n)/db(n) * da(n)/dz(n) * dL/da(n)
                # = 1 * actv'(z(n)) * dL/da(n)
                b_grad = np.sum(term_2_3, axis=1, keepdims=True) / batch_size

                # UPDATE/apply negative of average gradient change to biases / Update biases
                l2_term_for_b: np.ndarray = self.biases[i] * self.l2_lambda # Compute regularization term
                self.v_b[i] = self.m_beta * self.v_b[i] + (1 - self.m_beta) * (b_grad + l2_term_for_b)
                self.biases[i] += -1 * self.v_b[i] * self.lr

                if i == 0: continue # skip gradient descent calculation for input layer 
                # actual backpropagation
                # NOTE: a(n-1) affects all a(n), so backpropagation to a(n-1) will be related to all a(n)
                # dL/da(n-1)
                # = column-wise sum in w matrix [dz(n)/da(n-1) * da(n)/dz(n) * dL/da(n)]
                # = column-wise sum in w matrix [(w(n) * actv'(z(n)) * dL/da(n))]
                act_grad = np.matmul(self.weights[i].T, term_2_3)
                
                # important component for HIDDEN LAYER backpropagations
                # update for next layer
                # term_2_3 = da(n)/dz(n) * dL/da(n)
                term_2_3: np.ndarray = self._act_deriv_func[i-1](z_layers[i-1]) * act_grad
            
            p: float = (100.0 * (_+1) / epoch)
            print(f"Progress: [{'='*int(30*p/100):<30}] {_+1:>5} / {epoch} [{p:>6.2f}%]  ", end='\r')
        
        print("\n===== ===== Training Completed ===== =====")
