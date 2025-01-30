# Simple Neural Network

## Overview
This project is an intermediate level, fully functional neural network implemented from scratch in Python. Initially started out as a simple Python project to neural networks, it has gradually evolved into a more robust form.

White it is not exactly "simple" anymore, it is still a fun and *powerful* example of how a pile of Linear Algebra and Calculus can learn to predict.

First off, a huge shoutout to the awesome **NumPy** library, which is pretty much responsible for 99% of this neural network's speed.

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
- **Architecture**: Fully-connected (dense) layers with customizable layer sizes.
- **Activation functions**:
    - **Bounded**: Sigmoid, Tanh, Softmax.
    - **Unbounded**: ReLU, Leaky ReLU, PReLU (Learnable), Swish (Fixed/Learnable), Linear.
- **Loss functions**:
    - **Regression**: Mean-Squared Error (MSE).
    - **Multilabel classification**: Binary Cross-Entropy (BCE).
    - **Multiclass classification**: Categorial Cross-Entropy (CCE).
- **Training Algorithms**: Mini-batch gradient descent.
- **Techniques**: L2 (Ridge) regularization, Momentum.

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
    DenseLayer(4, 8, "leaky_relu"),
    DenseLayer(8, 6, "tanh"),
    DenseLayer(6, 6, "tanh"),
    DenseLayer(6, 2, "sigmoid")
])
```
#### Parameters
* `layers`: List of `DenseLayer` class.
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
        DenseLayer(4, 12, "prelu"),
        DenseLayer(12, 12, "prelu"),
        DenseLayer(12, 3, "sigmoid")
    ],
    loss_function="BCE",
    learn_rate=0.06,
    lambda_parem=0.001,
    momentum=0.75
)
```
```
Accuracy on 20,000 samples
Accuracy on each output:    96.48%   94.13%   93.05%
                   Expected |                   Predicted | Input Data
   1.0000   0.0000   0.0000 |    0.9947   0.0049   0.3137 |   1.856  0.199  2.490 -0.394
   0.0000   0.0000   1.0000 |    0.0000   0.0393   0.7646 |   4.646  0.327 -4.084  0.270
   1.0000   1.0000   1.0000 |    0.9999   1.0000   0.3026 |  -4.007 -2.221 -4.774  4.237
   0.0000   1.0000   0.0000 |    0.0000   0.9998   0.0002 |   4.895 -4.291 -5.255 -0.471
   1.0000   1.0000   1.0000 |    0.9982   0.9922   0.1835 |  -3.495 -0.052 -1.380 -0.636
   0.0000   1.0000   1.0000 |    0.0000   0.9042   0.7359 |  -3.088 -3.468  0.092 -5.644
   1.0000   1.0000   1.0000 |    0.0428   0.9936   0.8727 |  -2.144  1.603  1.352 -1.827
   0.0000   0.0000   0.0000 |    0.0000   0.1592   0.4527 |  -4.951 -1.259  5.049 -5.645
   0.0000   0.0000   0.0000 |    0.0232   0.0008   0.2139 |  -0.638 -2.186  5.770 -5.933
   0.0000   0.0000   1.0000 |    0.0000   0.2015   0.9998 |  -5.518 -3.723  3.072 -0.290
   1.0000   1.0000   0.0000 |    0.9439   1.0000   0.0001 |  -3.529 -4.184 -5.905 -5.933
   1.0000   1.0000   0.0000 |    1.0000   0.8075   0.0024 |  -4.899  2.757 -4.259 -1.695
   0.0000   1.0000   1.0000 |    0.0000   0.9351   0.9693 |  -5.476 -2.954  1.098 -2.479
   0.0000   0.0000   0.0000 |    0.0000   0.3218   0.1728 |   2.556 -4.684 -0.116  2.166
   0.0000   0.0000   1.0000 |    0.0000   0.2435   0.9926 |  -1.972 -5.750  1.202 -5.070
   0.0000   0.0000   1.0000 |    0.0000   0.0002   0.9999 |  -0.934 -3.311  4.627  2.496
```

### Multiclass Classification Performance
**Multiclass classification** is where each input belongs to exactly one class. This model uses **Softmax** activation in the output layer and **categorial cross-entropy (CCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        DenseLayer(4, 12, "swish"),
        DenseLayer(12, 12, "swish"),
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
Accuracy on each output:    95.71%   93.88%   97.85%
Overall categorization accuracy:    93.72%
                   Expected |                   Predicted | Input Data
   0.0000   1.0000   0.0000 |    0.0103   0.9896   0.0001 |   2.852  1.109  4.140  2.437
   0.0000   1.0000   0.0000 |    0.0008   0.9992   0.0000 |   2.209  1.098  5.309  2.773
   0.0000   1.0000   0.0000 |    0.0009   0.9991   0.0001 |   1.941  2.661  5.382  3.514
   0.0000   0.0000   1.0000 |    0.0183   0.0033   0.9784 |   1.373  2.080  0.535  0.962
   0.0000   0.0000   1.0000 |    0.0028   0.0013   0.9958 |   0.056  2.323  0.264  0.124
   1.0000   0.0000   0.0000 |    0.9907   0.0093   0.0000 |   3.824  1.586  1.699  3.606
   0.0000   0.0000   1.0000 |    0.0002   0.0808   0.9190 |   1.616  3.096  4.376  0.993
   1.0000   0.0000   0.0000 |    0.9548   0.0452   0.0000 |   3.931  1.158  2.399  3.718
   0.0000   0.0000   1.0000 |    0.0015   0.0002   0.9983 |   1.872  5.689  2.009  1.628
   0.0000   1.0000   0.0000 |    0.0027   0.9973   0.0000 |   3.257  1.502  5.378  3.265
   1.0000   0.0000   0.0000 |    0.9904   0.0096   0.0000 |   2.451  2.119  1.977  4.741
   0.0000   1.0000   0.0000 |    0.0036   0.1545   0.8419 |   0.949  2.988  3.346  1.810
   0.0000   0.0000   1.0000 |    0.0051   0.7565   0.2385 |   2.541  2.692  4.037  1.438
   0.0000   0.0000   1.0000 |    0.0012   0.0067   0.9921 |   1.741  4.470  3.773  1.731
   0.0000   0.0000   1.0000 |    0.0000   0.0000   1.0000 |   0.411  6.072  3.339  0.541
   0.0000   1.0000   0.0000 |    0.0007   0.9810   0.0184 |   1.190  1.896  4.510  1.874
```

### Regression Performance
**Regression** is where each input is mapped to a continuous value, rather than a discrete class. This model uses **Linear (Identity)** activation in the output layer and **mean-squared error (MSE)** loss for training.

*Note: Regression tasks generally require more rigorous data preparation and training control. Since output values can have a wider range compared to classification tasks, achieving high performance often demands more careful tuning of the model and training process.*
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        DenseLayer(4, 12, "prelu"),
        DenseLayer(12, 12, "prelu"),
        DenseLayer(12, 3, "id")
    ],
    loss_function="MSE",
    learn_rate=0.001, # Low learning rate to prevent exploding gradients by MSE loss
    lambda_parem=0.001,
    momentum=0.75
)
```
```
                   Expected |                   Predicted | Input Data
  -5.5640  -1.4503   3.6763 |   -3.2661  -2.5367   3.4250 |  -0.169 -0.895 -0.428  1.499
   0.7955   0.5794   5.8331 |    4.2458  -0.6110   4.5590 |   0.441 -0.128 -0.121  3.294
 -17.7009  -8.2832 -12.2048 |  -14.8624  -6.4282 -10.2248 |  -1.250 -2.950  0.945 -1.207
   6.2734   4.4161   1.1815 |    7.0831   3.6019  -2.2681 |   2.028 -2.053  2.231  3.033
  12.0378 -16.7046  -6.9762 |   10.7288 -15.0135  -6.0227 |  -0.292  1.560 -2.136  0.966
   3.2544  15.2852  11.5324 |    4.5163  12.2706  11.1467 |  -1.027  1.774  2.491  1.575
   2.6648  14.0585  13.4797 |   -0.7653   9.5942   8.6658 |  -2.766  2.665  3.172 -0.116
   1.2501   8.6527   4.1083 |    0.5697   7.2995   5.5298 |  -0.399  0.387  2.243  1.193
 -12.5605   7.2609   7.2117 |  -12.6458   7.9859   7.8587 |   1.863 -1.588 -1.892  0.595
  -1.7428  -5.2729  -3.1403 |    2.9982  -9.8791  -5.4485 |  -0.960 -0.276 -2.576 -2.978
 -13.6650  -5.8899  -3.0219 |  -12.9289  -5.4061  -2.2660 |  -2.826 -0.113  1.732  0.695
  -3.3439  -7.6137  -1.2843 |   -4.6730  -7.1900   0.5475 |  -2.362 -2.398 -1.096 -0.970
 -13.4230  -4.7750 -10.3038 |  -13.8468  -7.0792 -11.0188 |  -1.136 -1.926  1.967 -0.772
  18.7310 -26.3631 -14.0130 |   20.7680 -25.3665 -14.2681 |  -1.741  2.183 -2.729 -1.266
  12.0666  -6.7458  -3.7258 |   11.7687  -2.2814  -2.3433 |  -0.768  2.519  0.071 -1.166
  -7.5525  -2.5260  -4.9456 |   -7.0487  -1.8829  -1.9579 |  -1.409 -0.186  1.519 -0.349
```
As shown, the neural network performs exceptionally well on the synthetic data. If real-world data exhibits similar relationships between inputs and outputs, the network is likely to perform equally well.

## Synthetic Data Generation
Random data generation is used to simulate a variety of real-world scenarios where the relationship between input features and output targets is not immediately clear. By introducing randomization, we can create a diverse range of datasets that help test the network's ability to generalize and perform well across different situations.

This synthetic data helps in evaluating the model's performance without the need for a real-world dataset, which can be difficult to acquire or pre-process.

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

# Shape: (n, count of output features)
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

# Shape: (n, count of output features)
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

$$A(x)=x\sigma(\beta x)=\frac{x}{1+e^{-x}}$$

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