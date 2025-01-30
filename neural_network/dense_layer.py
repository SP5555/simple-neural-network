import numpy as np
from .activations import Activations
from .utils import Utils

class DenseLayer:
    def __init__(self, input_size: int, output_size: int, activation: str) -> None:
        Utils._act_func_validator(activation)

        self.act_func = Utils._get_act_func(activation)
        self.act_deriv_func = Utils._get_act_deriv_func(activation)
        self.learnable_deriv_func = Utils._get_learnable_alpha_grad_func(activation)

        # Learnable parameter bounds for each layer
        # defaults to (None, None, None) for non-learnable functions
        self._learnable_bounds = (
            Activations._learn_param_values.get(activation, (None, None, None))[1],
            Activations._learn_param_values.get(activation, (None, None, None))[2]
        )

        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2/input_size)
        self.biases = np.random.randn(output_size, 1)
                
        # extra learnable parameters
        # one per each neuron
        # learnable params of neurons in layers that don't use learnable parameters
        # will remain fixed at 1.0 throughout training
        init_value = Activations._learn_param_values.get(activation, (None,))[0] # default to None if not found
        self.alpha = np.full((output_size, 1), init_value)

        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.biases)
        self.v_alpha = np.zeros_like(self.alpha)

        self.w_grad = np.zeros_like(self.weights)
        self.b_grad = np.zeros_like(self.biases)
        self.alpha_grad = np.zeros_like(self.alpha)

        self.a = None
        self.z = None
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        
        # z = W*A + b
        self.z: np.ndarray = np.matmul(self.weights, input) + self.biases.reshape(-1, 1) # broadcasting
        # A = activation(z)
        self.a: np.ndarray = self._act_func(self.z, self.alpha)

        return self.a
    
    def backward(self, act_grad: np.ndarray, is_first: bool = False, is_final: bool = False) -> np.ndarray:
    
        # important component for LAST LAYER backpropagation
        # term_1_2 = dL/da(n) * da(n)/dz(n)
        if is_final and self.act_func.name == "softmax":

            da_wrt_dz = act_grad[:, :, None].transpose(1, 0, 2) # (batch_size, dim, 1)
            dL_wrt_da = self.act_deriv_func[-1](self.z, self.alpha) # Jacobians; (batch_size, dim, dim)
            
            t_1_2_3D = np.matmul(dL_wrt_da, da_wrt_dz) # (batch_size, dim, 1)
            term_1_2 = t_1_2_3D.squeeze(axis=-1).T # (dim, batch_size)
        else:
            term_1_2: np.ndarray = self.act_deriv_func[-1](self.z, self.alpha) * act_grad
        
        # BACKPROPAGATION
        
        batch_size = self.a.shape[1]

        # Math
        # z(n) = w(n)*a(n-1) + b(n)
        # a(n) = activation(z(n,learn_b)) # learnable parem is used only in some activations

        # CALCULATE derivative of loss with respect to weights
        # dL/dw(n)
        # = dL/da(n) * da(n)/dz(n) * dz(n)/dw(n)
        # = dL/da(n) * actv'(z(n)) * a(n-1)
        self.w_grad = np.matmul(term_1_2, self.a.T) / batch_size

        # CALCULATE derivative of loss with respect to biases
        # dL/db(n)
        # = dL/da(n) * da(n)/dz(n) * dz(n)/db(n)
        # = dL/da(n) * actv'(z(n)) * 1
        self.b_grad = np.sum(term_1_2, axis=1, keepdims=True) / batch_size

        if self.act_func.name in Activations._learnable_acts:
            # CALCULATE derivative of loss with respect to learnable    parameter
            # dL/dlearn_b(n)
            # = dL/da(n) * da(n)/dlearn_b(n)
            dL_wrt_dlearn_alpha = self.learnable_deriv_func(self.z, self.alpha) * act_grad
            self.alpha_grad = np.sum(dL_wrt_dlearn_alpha, axis=1, keepdims=True) / batch_size
        
        if not is_first:
            # actual backpropagation
            # NOTE: a(n-1) affects all a(n), so backpropagation to a(n-1) will be related to all a(n)
            # dL/da(n-1)
            # = column-wise sum in w matrix [dz(n)/da(n-1) * dL/da(n) * da(n)/dz(n)]
            # = column-wise sum in w matrix [w(n) * dL/da(n) * actv'(z(n))]
            act_grad = np.matmul(self.weights.T, term_1_2)
            return act_grad
        return

    def optimize(self, LR: float, l2_lambda: float, m_beta: float):
        # UPDATE/APPLY negative of average gradient change to weights
        l2_term_for_w: np.ndarray = self.weights * l2_lambda # Compute regularization term
        self.v_w = m_beta * self.v_w + (1 - m_beta) * (self.w_grad + l2_term_for_w)
        self.weights += -1 * self.v_w * LR

        # UPDATE/APPLY negative of average gradient change to biases
        l2_term_for_b: np.ndarray = self.biases * l2_lambda # Compute regularization term
        self.v_b = m_beta * self.v_b + (1 - m_beta) * (self.b_grad + l2_term_for_b)
        self.biases += -1 * self.v_b * self.LR

        if self.act_func.name in Activations._learnable_acts:
            # UPDATE/APPLY negative of average gradient change to learnable parameter
            l2_term_for_alpha: np.ndarray = self.alpha * l2_lambda # Compute regularization term
            self.v_alpha = m_beta * self.v_alpha + (1 - m_beta) * (self.alpha_grad + l2_term_for_alpha)
            self.alpha += -1 * self.v_alpha * LR
            self.alpha = np.clip(self.alpha, *self._learnable_bounds)