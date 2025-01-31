# Simple Neural Network

## Overview
This project started out as a simple Python experiment to explore neural networks. But, well, things escalated. Now, it's an almost fully-fledged, fully functional, intermediate-level neural network built completely from scratch.

White it is not exactly "simple" anymore, it is still a fun and *powerful* example of how a pile of numbers can learn to predict.

First off, a huge shoutout to the awesome **NumPy** library, because without it, this thing would be moving at the speed of a snail :snail:.

- [Features](#features)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
    - [Creating a Neural Network](#creating-a-neural-network)
    - [Training](#training)
    - [Utilities](#utilities)
- [Performance & Testing](#performance--testing)
    - [Multilabel](#multilabel-classification-performance)
    - [Multiclass](#multiclass-classification-performance)
    - [Regression](#regression-performance)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Experiment!](#experiment)
- [What to Expect?](#what-to-expect)
- [GPU Acceleration?](#gpu-acceleration)
- [Resources *& Inspirations :)*](#resources--inspirations-)

## Features
- **Architecture**: Feedforward Neural Network
- **Layers**:
    - **Dense Layer**: Fully-connected layer
    - **Dropout Layer**: Dense layer with dropout technique that drops a fraction of neurons during training, with an adjustable dropout rate
- **Activation functions**:
    - **Bounded**: Sigmoid, Tanh, Softmax
    - **Unbounded**: ReLU, Leaky ReLU, PReLU (Learnable), Swish (Fixed/Learnable), Linear
- **Loss functions**:
    - **Regression**: Mean-Squared Error (MSE)
    - **Multilabel classification**: Binary Cross-Entropy (BCE)
    - **Multiclass classification**: Categorial Cross-Entropy (CCE)
- **Training Algorithms**: Mini-batch gradient descent
- **Techniques**: L2 (Ridge) regularization, Momentum

## Dependencies
* **Python 3.0**
* **NumPy**

## Installation Instructions
Run the following commands to install it locally. Require `pip`.
```
git clone https://github.com/SP5555/simple-neural-network.git
cd simple-neural-network
pip install -r requirements.txt
```

## Usage

### Creating a Neural Network
To create a neural network with customizable layer configurations:
```python
# Example 1:
# A network with 4 input neurons, 6 hidden neurons, and 2 output neurons
# with leaky_relu activation in hidden layer and sigmoid activation in final layer
nn = NeuralNetwork(layers=[
    DenseLayer(4, 6, "leaky_relu"),
    DenseLayer(6, 2, "sigmoid")
])

# Example 2: A deeper network with multiple hidden layers
nn = NeuralNetwork(layers=[
    DenseLayer(4, 10, "leaky_relu"),
    DropoutLayer(10, 16, "tanh", 0.2),
    DropoutLayer(16, 14, "tanh", 0.2),
    DenseLayer(14, 2, "sigmoid")
])
```
#### Parameters
* `layers`: List of supported layer classes.
* `loss_function`: Loss function for training (E.g., `"MSE"`, `"BCE"`)
* `learn_rate`: Learning rate for gradient descent.
* `lambda_parem`: Regularization strength to prevent overfitting.
* `momentum`: Momentum to accelerate convergence.

Example with added parameters:
```python
nn = NeuralNetwork(
    layers=[
        DenseLayer(4, 12, "tanh"),
        DenseLayer(12, 12, "tanh"),
        DenseLayer(12, 3, "sigmoid")
    ],
    loss_function="BCE", # for multilabel classification
    learn_rate=0.05,
    lambda_parem=0.002,
    momentum=0.75
)
```
### Training
To train a network with input and output data:
```python
nn.train(
	input_list=input_train_list,
	output_list=output_train_list,
	epoch=1000,
	batch_size=64 # number of samples in each mini-batch for training
)
```
### Utilities
View the current weights and biases of the network:
```python
nn.utils.inspect_weights_and_biases()
```
Evaluate the accuracy of the network for classification tasks:
```python
nn.metrics.check_accuracy_classification(
	test_input=input_test_list,
	test_output=output_test_list
)
```
Compare and print out the predicted results with the desired results:
```python
nn.metrics.compare_predictions(input=data_i, output=data_o)
```

## Performance & Testing
This simple network delivers excellent results on basic regression and classification problems. Below is an example demonstrating its effectiveness.

The **synthetic** data (artificial data created using algorithms) is used to test the model's ability to predict outcomes based on the input features. All tests were conducted with **40,000 training datasets**. The performance is then evaluated on **20,000 unseen test datasets** generated using the same method (*the predictions for 16 unseen test datasets are compared with the expected outputs below*). [How is synthetic data generated?](#synthetic-data-generation)

### Multilabel Classification Performance
**Multilabel classification** is where each input can belong to multiple classes simultaneously. This model uses **Sigmoid** activation in the output layer and **binary cross-entropy (BCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        DenseLayer(4, 10, "tanh"),
        DropoutLayer(10, 16, "tanh", 0.1),
        DenseLayer(16, 12, "tanh"),
        DenseLayer(12, 3, "sigmoid")
    ],
    loss_function="BCE",
    learn_rate=0.04,
    lambda_parem=0.001,
    momentum=0.75
)
```
```
Accuracy on 20,000 samples
Accuracy on each output:    94.85%   91.62%   90.14%
                   Expected |                   Predicted | Input Data
   0.0000   1.0000   0.0000 |    0.0024   0.9470   0.0061 |  -0.863 -4.460 -0.880  0.671
   1.0000   0.0000   0.0000 |    0.9950   0.1269   0.0386 |  -0.976  5.639 -5.750 -4.498
   0.0000   0.0000   0.0000 |    0.0111   0.1051   0.7887 |  -1.345 -0.393  3.069 -4.314
   0.0000   0.0000   1.0000 |    0.0081   0.0492   0.9639 |  -2.221 -2.735  2.131  3.501
   0.0000   0.0000   1.0000 |    0.0056   0.1057   0.9038 |  -4.211 -2.141  2.535 -0.282
   1.0000   0.0000   1.0000 |    0.0346   0.0298   0.9808 |   3.172 -5.457  3.865 -1.221
   1.0000   0.0000   0.0000 |    0.9973   0.1439   0.0776 |  -0.453  3.657 -0.434 -2.179
   0.0000   0.0000   1.0000 |    0.0081   0.0400   0.9503 |  -2.034 -3.887  3.421  2.614
   0.0000   1.0000   0.0000 |    0.0039   0.9605   0.0161 |  -0.451 -5.107 -2.888  2.996
   1.0000   0.0000   0.0000 |    0.9962   0.0777   0.0739 |   2.044  4.571 -0.797 -1.706
   0.0000   1.0000   0.0000 |    0.0007   0.9004   0.0035 |   5.886 -4.976 -3.154  1.379
   0.0000   1.0000   0.0000 |    0.0032   0.9316   0.0117 |  -0.808 -4.195 -0.397  0.893
   1.0000   1.0000   0.0000 |    0.9970   0.6005   0.0304 |  -2.372  2.515 -1.659 -1.741
   0.0000   0.0000   1.0000 |    0.0341   0.0560   0.2474 |   5.888  0.516 -4.007  4.083
   0.0000   0.0000   0.0000 |    0.0423   0.0285   0.2642 |   4.762  2.259 -1.124  3.458
   0.0000   1.0000   0.0000 |    0.0009   0.9444   0.0053 |   2.849 -3.673 -5.073 -2.071
```

### Multiclass Classification Performance
**Multiclass classification** is where each input belongs to exactly one class. This model uses **Softmax** activation in the output layer and **categorial cross-entropy (CCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        DropoutLayer(4, 12, "prelu", 0.2),
        DropoutLayer(12, 16, "tanh", 0.2),
        DropoutLayer(16, 12, "swish", 0.2),
        DenseLayer(12, 3, "softmax")
    ],
    loss_function="CCE",
    learn_rate=0.02,
    lambda_parem=0.001,
    momentum=0.75
)
```
```
Accuracy on 20,000 samples
Accuracy on each output:    95.10%   94.39%   97.41%
Overall categorization accuracy:    93.46%
                   Expected |                   Predicted | Input Data
   1.0000   0.0000   0.0000 |    0.9557   0.0345   0.0098 |   2.204  1.824 -0.129  3.339
   0.0000   0.0000   1.0000 |    0.0429   0.0021   0.9550 |   2.016  4.399  1.124  1.736
   0.0000   0.0000   1.0000 |    0.0069   0.0023   0.9908 |   0.841  5.145  3.751  1.393
   0.0000   1.0000   0.0000 |    0.8770   0.1196   0.0034 |   3.525  2.609  3.751  4.873
   0.0000   0.0000   1.0000 |    0.0074   0.0051   0.9875 |   1.062  5.005  4.496  1.902
   1.0000   0.0000   0.0000 |    0.9624   0.0240   0.0136 |   3.905  3.749  0.882  3.813
   1.0000   0.0000   0.0000 |    0.8716   0.1246   0.0037 |   2.815  1.687  2.417  3.026
   0.0000   1.0000   0.0000 |    0.6804   0.3147   0.0050 |   1.484  2.607  3.639  3.861
   1.0000   0.0000   0.0000 |    0.8793   0.1172   0.0034 |   2.721  1.818  2.826  4.557
   1.0000   0.0000   0.0000 |    0.9558   0.0344   0.0098 |   3.533  3.269  1.796  4.631
   0.0000   1.0000   0.0000 |    0.0181   0.9813   0.0007 |   1.417  0.547  4.103  1.955
   1.0000   0.0000   0.0000 |    0.9623   0.0235   0.0141 |   3.746  4.672  3.064  4.457
   0.0000   1.0000   0.0000 |    0.0162   0.9735   0.0103 |   2.072  2.146  5.692  1.677
   1.0000   0.0000   0.0000 |    0.8955   0.1006   0.0039 |   2.976  1.378  1.735  3.287
   1.0000   0.0000   0.0000 |    0.9571   0.0326   0.0103 |   4.065  2.769  0.650  3.798
   0.0000   0.0000   1.0000 |    0.0069   0.0028   0.9903 |   0.618  3.582  2.431  0.763
```

### Regression Performance
**Regression** is where each input is mapped to a continuous value, rather than a discrete class. This model uses **Linear (Identity)** activation in the output layer and **mean-squared error (MSE)** loss for training.

*Note: Regression tasks generally require more rigorous data preparation and training control. Since output values can have a wider range compared to classification tasks, achieving high performance often demands more careful tuning of the model and training process.*
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        DenseLayer(4, 12, "prelu"),
        DropoutLayer(12, 16, "swish", 0.2),
        DropoutLayer(16, 12, "swish", 0.2),
        DenseLayer(12, 3, "id")
    ],
    loss_function="MSE",
    learn_rate=0.001,
    lambda_parem=0.001,
    momentum=0.75
)
```
```
                   Expected |                   Predicted | Input Data
 -11.7451  -7.9759  -8.5022 |  -10.5339  -1.9287  -1.9679 |   0.095 -2.955  1.829  2.178
   0.3964  -4.3869  -4.5237 |   -0.4985  -0.5781  -0.9001 |  -1.843  0.837  1.385 -2.296
  17.0592  12.9293  11.4285 |   14.2778  14.3896  11.8632 |   0.816  2.582  1.095  2.496
   5.9978 -18.7930 -14.2442 |    4.3040  -6.8387  -5.0272 |   1.598  2.640 -2.462 -0.579
  -4.3795  -3.0891  12.1098 |   -4.2830  -3.4058   6.7438 |  -0.769 -1.553 -2.956  1.892
  -5.6184 -10.8954 -10.3462 |    0.6998  -4.7540  -4.1112 |   0.904  1.029 -2.380 -2.480
   0.3074  -6.4738  -5.7237 |   -1.2189  -3.2559  -1.8537 |  -2.071  0.204  0.839 -1.542
  -4.3068   6.6485 -10.0870 |   -2.8991   1.9825  -0.2612 |   1.714 -1.868  2.569 -1.084
   6.4624 -11.2371  -0.9220 |    5.7389  -9.1423  -2.1587 |  -3.482  1.885  0.036  1.209
 -10.4068  -2.1054  -5.1978 |   -8.2743  -2.7459  -1.6978 |  -1.875 -2.806 -0.442 -2.326
 -12.6214   5.6478 -11.6833 |   -9.4515  -0.1268  -3.5569 |   0.877 -2.391  1.867 -2.344
  -2.3933  -5.3382   5.5390 |   -2.2661  -2.9847   5.0331 |  -1.178 -0.757 -1.066  3.064
   7.7869   9.3219   3.5277 |    9.2777   9.2739   7.2594 |   2.866  1.628 -0.076  0.838
  -4.7723   2.0477  -3.7942 |   -5.5643   0.5342  -3.0277 |  -0.129 -0.972  1.875  0.008
  12.5838  18.9219   6.5998 |   10.7665  11.3180   8.3046 |   2.215  2.538  1.293 -2.448
  -5.4142   2.6953   6.9392 |   -4.9911   0.7451   6.6858 |   2.814 -0.211 -2.071  2.182
```
As shown, the neural network performs exceptionally well on the synthetic data. If real-world data exhibits similar relationships between inputs and outputs, the network is likely to perform equally well.

## Synthetic Data Generation
With randomization, we can create diverse datasets that still follow a certain input-output relationship pattern to some extent (just like in the real world). This mimics the real world data and helps test the network's ability to generalize and perform well across different situations.

To note, we are NOT harvesting the full power of randomization as it would only generate complete gibberish. Instead, we establish a relation between inputs and outputs while introducing randomness to it at the same time. This synthetically generated data helps in evaluating the model's performance without the need for a real-world dataset, which can be difficult to acquire or pre-process.

*Monkey Language: The idea is, we define input-output relationships and generate data while heavily masking them with a reasonable level of randomness, making the patterns not immediately clear, even to humans. Then we evaluate the network's ability to cut through those random noise and uncover the underlying pattern.*

For the above example runs, 4 input features `i1, i2, i3, i4` and 3 output targets `o1, o2, o3` are generated as follows.
```python
# For multilabel classification demonstration
# Shape: (1, n)
i1 = np.random.uniform(-6, 6, size=n)
i2 = np.random.uniform(-6, 6, size=n)
i3 = np.random.uniform(-6, 6, size=n)
i4 = np.random.uniform(-6, 6, size=n)

# Shape: (1, n)
# Define arbitrary relationships between inputs and outputs for demonstration
o1 = (i1*i4 - 5*i2 < 2*i1*i3 - i4).astype(float)
o2 = (4*i1 - 2*i2*i3 + 0.4*i4*i2/i1 < -3*i3).astype(float)
o3 = (-i1/i4 + 0.3*i2 - 8*i2*i2/i3 < 2*i4).astype(float)

# Shape: (n, count of input features)
input_list = np.column_stack((i1, i2, i3, i4))

# Shape: (n, count of output targets)
output_list = np.column_stack((o1, o2, o3))
```
```python
# For multiclass classification demonstration
_input_features = 4
_output_classes = 3

input_list = np.zeros((n, _input_features))
output_list = np.zeros((n, _output_classes))

# (1, n) shape array of random class labels
class_labels = np.random.randint(0, _output_classes, size=n)

# Define input data ranges for each class
class_data = {
    0: [(2, 5), (1, 5), (0, 4), (3, 5)],
    1: [(1, 4), (1, 3), (3, 6), (1, 5)],
    2: [(0, 3), (2, 6), (0, 5), (0, 2)],
}

for c in range(_output_classes):
    # extract indices of class c
    indices = np.where(class_labels == c)[0]

    # generate/fill up data in input list
    for i, (low, high) in enumerate(class_data[c]):
        input_list[indices, i] = np.random.uniform(low, high, size=len(indices))
    # set correct class
    output_list[indices, c] = 1.0
```
```python
# For regression demonstration
# Shape: (1, n)
i1 = np.random.uniform(-3, 3, size=n)
i2 = np.random.uniform(-3, 3, size=n)
i3 = np.random.uniform(-3, 3, size=n)
i4 = np.random.uniform(-3, 3, size=n)

# Shape: (1, n)
# Define arbitrary relationships between inputs and outputs for demonstration
o1 = (i1*i4 + 5*i2 + 2*i1*i3 + i4)
o2 = (4*i1 + 2*i2*i3 + 0.4*i4*i2 + 3*i3)
o3 = (i1 + 0.3*i2 + 2*i3*i2 + 2*i4)

# Shape: (n, count of input features)
input_list = np.column_stack((i1, i2, i3, i4))

# Shape: (n, count of output targets)
output_list = np.column_stack((o1, o2, o3))
```
In all data generation, input features (`i1, i2, i3, i4, ...`) are exposed to some noise to better mimic real-world scenarios.
```python
# input noise
input_list = self._add_noise(input_list, noise=0.2)
```
```python
def _add_noise(self, data: np.ndarray, noise=0.5):
    # Add uniform noise element-wise to the entire NumPy array
    return data + np.random.uniform(-noise, noise, size=data.shape)
```

## Experiment!
Alright, there are *tons* of ways to mess this thing up. A neural network is not something that is strictly tailored to do one specific task, nor it is a random jumble of math that do awesome possum magic and surprisingly spits out excellent results. It all comes down to how *you* configure it. 

Feel free to mess around with the hyperparameters, or change things up entirely. You might discover some cool insights or get the network to do things it wasn't originally designed for. *What is it originally designed for anyway?*

*But hey, no one gets hurt if this thing breaks.*

## What to Expect?
This network has **insane level of scalibility** in terms of depth (number of layers) and width (neurons per layer), with the only real limit being your hardware, super impressive! Dense network architectures are suited the most for general-purpose tasks since they can approximate *any* mathematical function.

*Monkey Language: We're saying that stacks of linear equations and activation functions can approximate even the most complicated mathematical relations to some extent, similar to how taylor series approximation works. As long as there's an underlying mathematical pattern between the input and output, this network can learn to model it!*

Meanwhile, due to the **Lack of Structure Awareness**, the input is just a big flat vector of numbers to this network. And therefore, it can NOT understand the spatial hierarchies like **Convolutional Neural Networks (CNN)** and sequential dependencies like **Recurrent Neural Networks (RNN)**.

*Monkey Language: this means, you can't train this dense network to recognize images, or "understand" the meaning of texts and time-series data efficiently.*

That said, this does not mean it's impossible, just less efficient. **CNN**s and **RNN**s just happen to have "specialized brains" for those tasks. This network might feel like it's "brute-forcing" solutions instead of leveraging patterns more naturally.

Interestingly, **CNN**s and **RNN**s still end with a **fully connected (dense) neural network** for final predictions. This means only one thing: this **Simple Neural Network** can be extended in the future with **CNN**s or **RNN**s.

## GPU Acceleration?
This neural network implementation is built using pure **NumPy** utilities, which allows for easy conversion to **CuPy** for GPU acceleration without changing the code structure. By simply replacing **NumPy** with **CuPy**, computations can be offloaded to a CUDA-capable GPU, for faster training for large network and datasets.

However, using GPU acceleration may introduce significant overhead if the neuron count is low or the network has only a few layers, where the benefits of GPU processing may be outweighed by the cost of transferring data to the GPU.

## Resources *& Inspirations :)*
- 3Blue1Brown: [Neural Network Series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=856ZSYGlqoSCdHB2)
- GeeksForGeeks: [What is a Neural Network?](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/)
- TensorFlow: [A Neural Network Playground](https://playground.tensorflow.org/)
- StatQuest with Josh Starmer: [The Softmax Derivative](https://youtu.be/M59JElEPgIg?si=S_ERldGE5K5Jib0E)
- [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)

## Future Improvements
- chill out. Not right now.

# Abyssal Zone (Recommended to not proceed)
You made it this far. A scrolling enthusiast. Let's see where this takes you.

## Activation Functions
Activation functions calculate the output value of a neuron given the input value. As simple as that. They are inspired by the the behavior of neurons in the brain. A biological neuron "fires" or activates when it receives a signal exceeding a certain threshold. Similarly, activation functions in artificial neurons produce outputs based on the input values. This allows artificial neurons in a neural network layer to process information in a way that mimics biological neurons, deciding whether to pass on a signal to the next layer or not depending on the strength of the input.

### Sigmoid
An 'S' shaped function very commonly used in neural networks. Since output ranges between 0 and 1, **Sigmoid** is useful for classification tasks. Suffers from **vanishing gradient problem**.

*Monkey Language: If* $x$  *becomes too small (large negative) or too large (large positive), gradients (derivatives) used to update the network become very small, making the network learn very slowly, if at all.*

$$A(x)=\sigma ( x)=\frac{1}{1+e^{-x}}$$

### Tanh
Classic Hyperbolic Tangent function. **Tanh** generally gives better performance than **Sigmoid** when used in hidden layers due to its ability to express both negative and positive numbers from -1 to 1. Therefore, useful for classification. This one suffers from **vanishing gradient problem** too.

$$A( x) =\tanh( x) =\frac{e^{x} -e^{-x}}{e^{x} +e^{-x}}$$

### ReLU
The Rectified Linear Unit activation function is an unbounded function with output ranging up to infinity. Main strength lies in its simplicity not having to do nasty math functions on the silicon. However, **ReLU** suffers from **dying ReLU problem**.

*Monkey Language: when input* $x$  *is negative, it basically cuts off information or kills the neurons with the output activation value of 0. Therefore, backpropagation algorithms can't pass through this dead gradient barrier.*

$$A(x) =\begin{cases}
x & \text{if}\ x >0\\
0 & \text{if}\ x\leqslant 0
\end{cases}$$

### Leaky ReLU
Basically **ReLU** but leaking on the negative side. **Leaky ReLU** can mitigate the **dying ReLU problem**. By introducing a small slope, $\alpha$ (usually 0.01, can be configured), on the negative side, it creates non-zero gradient for negative inputs, potentially allowing backpropagation to revive the "dead" neurons which **ReLU** can't. **Leaky ReLU** is computationally cheap like **ReLU**, making it a cost-efficient activation function for regression problems.

$$A(x) =\begin{cases}
x & \text{if}\ x >0\\
\alpha x & \text{if}\ x\leqslant 0
\end{cases}$$

### PReLU (Parametric ReLU)
Disturbed by the fixed slope $\alpha$ in **Leaky ReLU**? This is where **PReLU** comes in. Here, $\alpha$ becomes a *learnable* variable which means it is also updated during backpropagation. With each neuron learning its own negative slope, the network becomes more flexible and capable of better performance. However, this comes at a small computational cost, as the learnable $\alpha$ introduces additional parameters to optimize.

### Linear (Identity)
For those who love simplicity. It's lightning-fast since it involves literally no math. However, due to the lack of non-linearity, a deep network made up of thousands, millions, or even billions of layers with **Linear** activations is mathematically equivalent to a single linearly activated layer. Too bad right? Yeah, this is why **Linear** is only used in the final layer when raw values are required (regression tasks). No point using this in hidden layers.

$$A(x)=x$$

### Swish
When sharp changes in gradient of **ReLU** variants do not fit your favor and you still want the power of **ReLU**, **Swish** comes to the rescue. Thanks to its smooth gradient, it can outperform traditional **ReLU** variants. What makes it even more powerful? The addition of a learnable parameter, $\beta$, ensuring a high performance uplift. Of course, with great power comes great cost, **Swish** is by far the most computationally expensive activation to calculate. Alternatively, learnable parameter $\beta$ can be disabled (permanently set to 1) to reduce computation while maintaining relatively high performance.

$$A(x)=x\sigma(\beta x)=\frac{x}{1+e^{-\beta x}}$$

### Softmax
Unlike all other activation functions, **Softmax** squashes raw scores of all neurons in same layer into a probability distribution, ensuring all values are positive and sum up to 1. Backpropagation and gradient calculations become further complicated due to its interconnectivity between activations in the same layer. Since all activations sum up to just 1, using **Softmax** in hidden layers would effectively suppress weaker activated neurons and exaggerate stronger ones, which is not what we want. For this reason, **Softmax** is typically reserved for the final layer in multiclass classification tasks.

```math
A_{i}(x) =\frac{e^{x_{i}}}{\sum_{j=1}^{n} e^{x_{j}}}
```

## Loss Functions
Loss functions measure how well a model's predictions align with the target values. They get applied to each neuron in the output layer, and the whole point of training a neural network is to adjust the parameters so that the loss gets as small as possible. The better we can get each neuron to predict with less loss, the more accurate the model becomes. The entire realm of Linear Algebra and Calculus has been summoned to minimize this loss value.

### Mean-Squared Error (MSE)
A commonly used loss function. As the name suggests, it is just the square of the error. The squaring effect diminishes small errors (values less than 1) but causes large errors to explode, making this both a strength and a weakness of the **Mean-Squared Error** loss function. Due to its ability to deal with errors of all sizes (assuming we have infinite hardware power to handle extremely large squared values), it is most suited for regression tasks.

$$MSE=( y_{actual} -y_{predicted})^{2}$$

### Binary Cross-Entropy (BCE)
When **MSE** struggles with small error values (less than 1) due to squaring, which diminishes those values, **Binary Cross-Entropy** is the answer. The logarithmic functions help amplify the error between 0 and 1, making it much more effective at penalizing incorrect predictions in this tiny range. This gives **BCE** an edge, especially in binary classification tasks where the model's predictions need to be confident (either 0 or 1).

$$BCE=-y_{actual}\log(y_{predicted})-(1-y_{actual})\log(1-y_{predicted})$$

### Categorial Cross-Entropy (CCE)
While **BCE** penalizes both 1s for not being 0 and 0s for not being 1, **Categorical Cross-Entropy** (CCE) focuses only on punishing the 0s when they should be 1. Designed to work with one-hot encoded values *together* with the **Softmax** activation function, **CCE** can guide a neural network to categorize data accurately by heavily penalizing incorrect 0 predictions. Once low raw scores become high enough, the **Softmax** activation will convert them into a high probability distribution.

$$CCE=-y_{actual}\log(y_{predicted})$$

## Optimization Techniques

### L2 (Ridge) Regularization
Just like in life, being an extremist is never good. The same goes for neural networks. Both underfitting and overfitting are problems to avoid:

- **Underfitting** happens when a network isn't able to learn well enough. This can happen due to various factors like a low neuron count, lack of non-linearity, or too low of a learning rate.

*Monkey Language: The network can't learn because it does not have enough brain cells or its brain cells are too simple.*
- **Overfitting** occurs when the network performs too perfectly on the training data, but fails to predict unseen data well. Essentially, the network "memorizes" the data instead of learning the underlying pattern. This usually happens due to overtraining, overwhelming number of parameters, overly strong gradients, or small datasets, etc.

*Monkey Language: The network memorized the material taught in class without understanding it, only to fail the exam when unseen questions come up.*

Rather than trying to balance everything from the start, we begin with a highly capable neural network. Possibly, multiple layers and appropriate activation functions. Then, we introduce a bit of interference in the learning process to prevent the network from learning too perfectly. This is exactly what **L2 (Ridge) Regularization** does.

$$Regularized\ Loss=Loss+\frac{1}{2} \lambda w^{2}$$

What is the intuition? In neural networks, large parameter values have a stronger influence on the output. If these large parameters begin to "memorize" the given data, the **L2 regularization** term, $\frac{1}{2} \lambda w^{2}$ adds to the overall loss and heavily penalizes them with the squared values. This makes stronger parameters decay more quickly back toward zero. In other words, it is like a void pulling all parameters toward zero, ensuring that no value in the network explodes into huge negatives or positives. This process thereby reduces the risk of "memorizing" the data and helps the network generalize better. The $\lambda$ controls the strength of regularization.

### Momentum
In physics, when an object is moving, it has momentum. Objects with momentum continue to maintain that momentum unless external forces act upon them. Similarly, in neural networks, when parameters are moving toward a local minimum to minimize the loss, the momentum technique gives them the ability to "glide." This "gliding" helps them escape high-loss plateaus faster, allowing them to reach the local minimum more efficiently. It introduces the concept of "velocity" for each parameter in the model. The velocity is updated as follows:

$$v_{t+1} =\beta v_{t} +( 1-\beta ) \nabla L( w_{t})$$

where $\beta$ is the momentum strength, ranging between 0 and 1. A value of 0 disables the "gliding" behavior while 1 results in full "gliding" with no update. Ideally, $\beta$ is set between 0.5 and 0.8.

And the calculated velocity is applied to parameters as follows:

$$w_{t+1} =w_{t} -\alpha v_{t+1}$$

Here, $\alpha$ is simply the **learn rate**, which controls how fast or slow the velocity moves the parameter. Ideally, $\alpha$ values less than 0.01 are preferred. It can go as low as 0.0001 in some cases.