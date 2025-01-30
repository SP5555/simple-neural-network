import numpy as np
from .activations import Activations
from .exceptions import InputValidationError
from .utils import Utils

class DenseLayer:
    
    def __init__(self, input_size: int, output_size: int, activation: str) -> None:
        if input_size == 0:
            raise InputValidationError("A layer can't have 0 input.")
        if output_size == 0:
            raise InputValidationError("A layer can't have 0 output (0 neurons).")
        activation = activation.strip().lower()
        Utils._act_func_validator(activation)
        
        self.input_size = input_size
        self.output_size = output_size
        self.act_name = activation

    def build(self, is_first: bool = False, is_final: bool = False) -> None:

        # helpers, they save a lot of power
        self._is_first = is_first
        self._is_final = is_final

        if not is_final and self.act_name in Activations._LL_exclusive:
            raise InputValidationError(f"{self.act_name} activation can't be used in hidden layers.")

        self._act_func = Utils._get_act_func(self.act_name)
        self._act_deriv_func = Utils._get_act_deriv_func(self.act_name)
        self._learnable_deriv_func = Utils._get_learnable_alpha_grad_func(self.act_name)

        # Learnable parameter bounds for each layer
        # defaults to None for non-learnable functions
        bounds = Activations._learn_param_values.get(self.act_name, None)
        self._learnable_bounds = (bounds[1], bounds[2]) if bounds else None

        self.weights = np.random.randn(self.output_size, self.input_size) * np.sqrt(2/self.input_size)
        self.biases = np.random.randn(self.output_size, 1)
                
        # extra learnable parameters
        # one per each neuron
        # learnable params of neurons in layers that don't use learnable parameters
        # will remain fixed at 1.0 throughout training
        init_value = Activations._learn_param_values.get(self.act_name, (None,))[0] # default to None if not found
        self.alpha = np.full((self.output_size, 1), init_value)

        # inialized with None for efficiency
        # gradients
        self._w_grad = None          # same shape as self.weights
        self._b_grad = None          #      //       self.biases
        self._alpha_grad = None      #      //       self.alpha

        # velocities
        self._v_w = np.zeros_like(self.weights)
        self._v_b = np.zeros_like(self.biases)
        self._v_alpha = np.zeros_like(self.alpha)

        # raw input, raw output and activation
        self._a_in = None            # shape: (input_size, batch_size)
        self._z = None               # shape: (output_size, batch_size)
        self._a = None               # shape: (output_size, batch_size)
    
    # compute a layer's output based on the input.
    def forward(self, input: np.ndarray) -> np.ndarray:
        
        self._a_in: np.ndarray = input
        # z = W*A + b
        self._z: np.ndarray = np.matmul(self.weights, self._a_in) + self.biases.reshape(-1, 1) # broadcasting
        # A = activation(z)
        self._a: np.ndarray = self._act_func(self._z, self.alpha)

        return self._a
    
    # computes gradients for weights, biases, alpha, and inputs based on the loss.
    # returns the input gradient to be used in the previous layer's backward pass.
    def backward(self, act_grad: np.ndarray) -> np.ndarray:
    
        # important component for backpropagation
        # term_1_2 = dL/da(n) * da(n)/dz(n)
        #          = dL/da(n) * actv'(z(n))

        # Softmax case is different
        if self._act_func.name == "softmax":

            da_wrt_dz = act_grad[:, :, None].transpose(1, 0, 2) # (batch_size, dim, 1)
            dL_wrt_da = self._act_deriv_func(self._z, self.alpha) # Jacobians; (batch_size, dim, dim)
            
            t_1_2_3D = np.matmul(dL_wrt_da, da_wrt_dz) # (batch_size, dim, 1)
            term_1_2 = t_1_2_3D.squeeze(axis=-1).T # (dim, batch_size)
        else:
            # alpha is broadcasted to match z dimensions
            alpha_extended = np.where(self.alpha == None, 1.0, self.alpha) * np.ones_like(self._z)
            term_1_2: np.ndarray = self._act_deriv_func(self._z, alpha_extended) * act_grad
   
        # BACKPROPAGATION
        
        batch_size = self._a.shape[1]

        # Math
        # z(n) = w(n)*a(n-1) + b(n)
        # a(n) = activation(z(n,learn_b)) # learnable parem is used only in some activations

        # CALCULATE derivative of loss with respect to weights
        # dL/dw(n)
        # = dL/da(n) * da(n)/dz(n) * dz(n)/dw(n)
        # = dL/da(n) * actv'(z(n)) * a(n-1)
        self._w_grad = np.matmul(term_1_2, self._a_in.T) / batch_size

        # CALCULATE derivative of loss with respect to biases
        # dL/db(n)
        # = dL/da(n) * da(n)/dz(n) * dz(n)/db(n)
        # = dL/da(n) * actv'(z(n)) * 1
        self._b_grad = np.sum(term_1_2, axis=1, keepdims=True) / batch_size

        if self.act_name in Activations._learnable_acts:
            # CALCULATE derivative of loss with respect to learnable    parameter
            # dL/dlearn_b(n)
            # = dL/da(n) * da(n)/dlearn_b(n)
            dL_wrt_dlearn_alpha = self._learnable_deriv_func(self._z, self.alpha) * act_grad
            self._alpha_grad = np.sum(dL_wrt_dlearn_alpha, axis=1, keepdims=True) / batch_size

        # first layer does not have previous layer
        # not need to return input gradient   
        if self._is_first: return

        # actual backpropagation
        # NOTE: a(n-1) affects all a(n), so backpropagation to a(n-1) will be related to all a(n)
        # dL/da(n-1)
        # = column-wise sum in w matrix [dz(n)/da(n-1) * dL/da(n) * da(n)/dz(n)]
        # = column-wise sum in w matrix [w(n) * dL/da(n) * actv'(z(n))]
        act_grad = np.matmul(self.weights.T, term_1_2)
        return act_grad

    # updates the parameters (weights, biases, alpha) based on gradients calculated by backward().
    def optimize(self, LR: float, l2_lambda: float, m_beta: float):
        # UPDATE/APPLY negative of average gradient change to weights
        l2_term_for_w: np.ndarray = self.weights * l2_lambda # Compute regularization term
        self._v_w = m_beta * self._v_w + (1 - m_beta) * (self._w_grad + l2_term_for_w)
        self.weights += -1 * self._v_w * LR

        # UPDATE/APPLY negative of average gradient change to biases
        l2_term_for_b: np.ndarray = self.biases * l2_lambda # Compute regularization term
        self._v_b = m_beta * self._v_b + (1 - m_beta) * (self._b_grad + l2_term_for_b)
        self.biases += -1 * self._v_b * LR

        if self.act_name in Activations._learnable_acts:
            # UPDATE/APPLY negative of average gradient change to learnable parameter
            l2_term_for_alpha: np.ndarray = self.alpha * l2_lambda # Compute regularization term
            self._v_alpha = m_beta * self._v_alpha + (1 - m_beta) * (self._alpha_grad + l2_term_for_alpha)
            self.alpha += -1 * self._v_alpha * LR
            self.alpha = np.clip(self.alpha, *self._learnable_bounds)
    
    def _get_param_count(self) -> int:
        w = self.input_size * self.output_size
        s = self.output_size
        lp = 0
        if self.act_name in Activations._learnable_acts:
            lp = self.output_size
        return w + s + lp