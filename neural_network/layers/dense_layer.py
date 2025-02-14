import numpy as np
from ..activations.activation import Activation
from ..exceptions import InputValidationError
from .layer import Layer

class DenseLayer(Layer):
    """
    A fully connected (dense) layer where every input neuron 
    is connected to every output neuron.

    Parameters
    ----------
    input_size : int
        Number of input neurons. Must match the output size of the previous layer 
        or the input dimension if this is the first layer.
    
    output_size : int
        Number of output neurons. Must match the input size of the next layer 
        or the final output dimension if this is the last layer.
    
    activation : Activation
        Activation function to apply to the output neurons.
    
    weight_decay : float, optional
        Strength of L2 regularization.
        Default is 0.0, meaning no regularization.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Activation,
                 weight_decay: float = 0.0) -> None:

        super().__init__(input_size, output_size, activation, weight_decay)

    def build(self, is_first: bool = False, is_final: bool = False) -> None:

        # helpers, they save a lot of power
        self._is_first = is_first
        self._is_final = is_final

        if not is_final and self.activation.is_LL_exclusive:
            raise InputValidationError(f"{self.activation.__class__.__name__} activation can't be used in hidden layers.")

        self.weights = np.random.randn(self.output_size, self.input_size) * np.sqrt(2/self.input_size)
        self.biases = np.random.randn(self.output_size, 1)
 
        # initialized with None for efficiency
        # gradients
        self._w_grad = None     # same shape as self.weights
        self._b_grad = None     #      //       self.biases

        # raw input, raw output and activation
        self._a_in = None       # shape: (input_size, batch_size)
        self._z = None          # shape: (output_size, batch_size)
        self._a = None          # shape: (output_size, batch_size)
    
    # compute a layer's output based on the input.
    def forward(self, input: np.ndarray) -> np.ndarray:
        
        self._a_in: np.ndarray = input
        # z = W*A_in + b
        self._z: np.ndarray = np.matmul(self.weights, self._a_in) + self.biases # auto-broadcasting
        # A_out = activation(z, learn_b)
        self.activation.build_expression(self._z)
        self._a: np.ndarray = self.activation.forward()

        return self._a
    
    # computes gradients for weights, biases, alpha, and inputs based on the loss.
    # returns the input gradient which is required for the previous layer's backward pass.
    # basically BACKPROPAGATION
    def backward(self, act_grad: np.ndarray) -> np.ndarray:

        batch_size = self._a.shape[1]

        # important component for backpropagation
        # term_1_2 = dL/da(n) * da(n)/dz(n)
        #          = dL/da(n) * actv'(z(n))

        # Softmax case is different
        if self.activation.__class__.__name__ == "Softmax":

            da_wrt_dz = act_grad[:, :, None].transpose(1, 0, 2) # (batch_size, out, 1)
            dL_wrt_da = self.activation.backward(self._z) # Jacobians; (batch_size, out, out)
            
            t_1_2_3D = np.matmul(dL_wrt_da, da_wrt_dz) # (batch_size, out, 1)
            term_1_2 = t_1_2_3D.squeeze(axis=-1).T # (out, batch_size)
        else:
            # alpha is broadcasted to match z dimensions automatically in _act_deriv_func
            # alpha_extended = np.where(self.alpha == None, 1.0, self.alpha) * np.ones_like(self._z)
            # matrix dims: (out, batch_size) = [(out, batch_size) ele-wise-opt (out, batch_size)] ele-wise-opt-mult (out, batch_size)
            term_1_2: np.ndarray = self.activation.backward() * act_grad

        # Math
        # z(n) = w(n)*a(n-1) + b(n)
        # a(n) = activation(z(n, learn_b)) # learnable parem is used only in some activations

        # CALCULATE derivative of loss with respect to weights
        # dL/dw(n)
        # = dL/da(n) * da(n)/dz(n) * dz(n)/dw(n)
        # = dL/da(n) * actv'(z(n)) * a(n-1)
        # matrix dims: (out, in) = (out, batch_size) * (batch_size, in)
        self._w_grad = np.matmul(term_1_2, self._a_in.T) / batch_size
        l2_term_for_w: np.ndarray = self.weights * self.L2_lambda # Compute regularization term
        self._w_grad += l2_term_for_w

        # CALCULATE derivative of loss with respect to biases
        # dL/db(n)
        # = dL/da(n) * da(n)/dz(n) * dz(n)/db(n)
        # = dL/da(n) * actv'(z(n)) * 1
        # matrix dims: (out, 1) = squash-add along axis 1 (out, batch_size)
        self._b_grad = np.sum(term_1_2, axis=1, keepdims=True) / batch_size
        l2_term_for_b: np.ndarray = self.biases * self.L2_lambda # Compute regularization term
        self._b_grad += l2_term_for_b

        if self.activation.is_learnable:
            # CALCULATE derivative of loss with respect to learnable parameter
            # dL/dlearn_b(n)
            # = dL/da(n) * da(n)/dlearn_b(n)
            # matrix dims: (out, batch_size) = (out, batch_size) ele-wise-opt (out, batch_size)
            dL_wrt_dlearn_alpha = self.activation.get_param_grad() * act_grad
            # matrix dims: (out, 1) = squash-add along axis 1 (out, batch_size)
            self.activation.alpha_grad = np.sum(dL_wrt_dlearn_alpha, axis=1, keepdims=True) / batch_size
            l2_term_for_alpha: np.ndarray = self.activation.alpha * self.L2_lambda # Compute regularization term
            self.activation.alpha_grad += l2_term_for_alpha

        # first layer does not have previous layer
        # not need to return input gradient   
        if self._is_first: return

        # actual backpropagation
        # NOTE: a(n-1) affects all a(n), so backpropagation to a(n-1) will be related to all a(n)
        # dL/da(n-1)
        # = column-wise sum in w matrix [dz(n)/da(n-1) * dL/da(n) * da(n)/dz(n)]
        # = column-wise sum in w matrix [w(n) * dL/da(n) * actv'(z(n))]
        # matrix dims: (in, batch_size) = (in, out) * (out, batch_size)
        act_grad = np.matmul(self.weights.T, term_1_2)
        return act_grad

    def _get_params(self) -> list[dict]:
        params = [
            {'weight': self.weights, 'grad': self._w_grad},
            {'weight': self.biases, 'grad': self._b_grad}
        ]
        if self.activation.is_learnable:
            params.append({
                'weight': self.activation.alpha,
                'grad': self.activation.alpha_grad,
                'learnable': True,
                'constraints': self.activation.alpha_constraints
            })
        return params
    
    def _get_param_count(self) -> int:
        w = self.input_size * self.output_size
        s = self.output_size
        lp = self.output_size if self.activation.is_learnable else 0
        return w + s + lp