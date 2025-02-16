# Activation Functions
Activation functions calculate the output value of a neuron given the input value. As simple as that. They are inspired by the the behavior of neurons in the brain. A biological neuron "fires" or activates when it receives a signal exceeding a certain threshold. Similarly, activation functions in artificial neurons produce outputs based on the input values. This allows artificial neurons in a neural network layer to process information in a way that mimics biological neurons, deciding whether to pass on a signal to the next layer or not depending on the strength of the input.

## Sigmoid
An 'S' shaped function very commonly used in neural networks. Since output ranges between 0 and 1, **Sigmoid** is useful for classification tasks. Suffers from **vanishing gradient problem**.

*Monkey Language: If* $x$  *becomes too small (large negative) or too large (large positive), gradients (derivatives) used to update the network become very small, making the network learn very slowly, if at all.*

$$A(x)=\sigma ( x)=\frac{1}{1+e^{-x}}$$

## Tanh
Classic Hyperbolic Tangent function. **Tanh** generally gives better performance than **Sigmoid** when used in hidden layers due to its ability to express both negative and positive numbers from -1 to 1. Therefore, useful for classification. This one suffers from **vanishing gradient problem** too.

$$A( x) =\tanh( x) =\frac{e^{x} -e^{-x}}{e^{x} +e^{-x}}$$

## ReLU
The Rectified Linear Unit activation function is an unbounded function with output ranging up to infinity. Main strength lies in its simplicity not having to do nasty math functions on the silicon. However, **ReLU** suffers from **dying ReLU problem**.

*Monkey Language: when input* $x$  *is negative, it basically cuts off information or kills the neurons with the output activation value of 0. Therefore, backpropagation algorithms can't pass through this dead gradient barrier.*

$$A(x) =\begin{cases}
x & \text{if}\ x >0\\
0 & \text{if}\ x\leqslant 0
\end{cases}$$

## Leaky ReLU
Basically **ReLU** but leaking on the negative side. **Leaky ReLU** can mitigate the **dying ReLU problem**. By introducing a small slope, $\alpha$ (usually 0.01, can be configured), on the negative side, it creates non-zero gradient for negative inputs, potentially allowing backpropagation to revive the "dead" neurons which **ReLU** can't. **Leaky ReLU** is computationally cheap like **ReLU**, making it a cost-efficient activation function for regression problems.

$$A(x) =\begin{cases}
x & \text{if}\ x >0\\
\alpha x & \text{if}\ x\leqslant 0
\end{cases}$$

## Parametric ReLU (PReLU)
Disturbed by the fixed slope $\alpha$ in **Leaky ReLU**? This is where **PReLU** comes in. Here, $\alpha$ becomes a *learnable* variable which means it is also updated during backpropagation. With each neuron learning its own negative slope, the network becomes more flexible and capable of better performance. However, this comes at a small computational cost, as the learnable $\alpha$ introduces additional parameters to optimize.

## Linear (Identity)
For those who love simplicity. It's lightning-fast since it involves literally no math. However, due to the lack of non-linearity, a deep network made up of thousands, millions, or even billions of layers with **Linear** activations is mathematically equivalent to a single linearly activated layer. Too bad right? Yeah, this is why **Linear** is only used in the final layer when raw values are required (regression tasks). No point using this in hidden layers.

$$A(x)=x$$

## Softplus (Smooth ReLU)
When sharp changes in the gradient of **ReLU** variants do not fit your favor and you still want the power of **ReLU**, **Softplus** comes to the rescue. Thanks to its smooth gradient, it can outperform traditional **ReLU** variants in certain scenarios.

$$A( x) =\ln\left( 1+e^{x}\right)$$

## Swish
Want to add some sprinkle of learnable parameter to **Softplus**-like activation? **Swish** is here. With $\beta$, ensuring a performance uplift (again, in certain scenarios). Of course, with great power comes great cost, **Swish** is by far the most computationally expensive activation to calculate. Alternatively, learnable parameter $\beta$ can be disabled (permanently set to 1) to reduce computation while maintaining relatively high performance.

$$A(x)=x\sigma(\beta x)=\frac{x}{1+e^{-\beta x}}$$

## Softmax
Unlike all other activation functions, **Softmax** squashes raw scores of all neurons in same layer into a probability distribution, ensuring all values are positive and sum up to 1. Backpropagation and gradient calculations become further complicated due to its interconnectivity between activations in the same layer. Since all activations sum up to just 1, using **Softmax** in hidden layers would effectively suppress weaker activated neurons and exaggerate stronger ones, which is not what we want. For this reason, **Softmax** is typically reserved for the final layer in multiclass classification tasks.

```math
A_{i}(x) =\frac{e^{x_{i}}}{\sum_{j=1}^{n} e^{x_{j}}}
```