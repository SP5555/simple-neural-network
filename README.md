# Simple Neural Network

## Overview
This project started out as a simple Python experiment to explore neural networks. But, well, things escalated. Now, it's a fully functional, intermediate-level neural network built completely from scratch.

White it is not exactly "simple" anymore, it is still a fun and *powerful* example of how a pile of numbers can learn to predict (*better than humans*).

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
    - **Bounded**: Sigmoid, Softmax, Tanh
    - **Unbounded**: Linear, ReLU, Leaky ReLU, PReLU (Learnable), Swish (Fixed/Learnable)
- **Loss functions**:
    - **Regression**: Mean-Squared Error (MSE), Mean-Absolute Error (MAE), Huber Loss
    - **Multilabel classification**: Binary Cross-Entropy (BCE)
    - **Multiclass classification**: Categorial Cross-Entropy (CCE)
- **Training Algorithms**: Mini-batch gradient descent
- **Optimizers**: Adam, AdaGrad, Momentum, RMSprop, Stochastic Gradient Descent (SGD)

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
#### Parameters
* `layers`: List of supported layer classes.
* `loss_function`: Loss function for training (E.g., `MSE`, `BCE`)
* `optimizer`: an instance of a derived optimizer class (E.g., `SGD`, `Momentum`)
```python
# Example 1:
# A network with 4 input neurons, 6 hidden neurons, and 2 output neurons
# leaky_relu activation in hidden layer and sigmoid activation in final layer, SGD optimizer and BCE Loss function
nn = NeuralNetwork(layers=[
        DenseLayer(4, 6, "leaky_relu"),
        DenseLayer(6, 2, "sigmoid"),
    ],
    loss_function=BCE(),
    optimizer=SGD(learn_rate=0.02)
)

# Example 2: with decaying rates
nn = NeuralNetwork(
    layers=[
        DenseLayer(4, 12, "prelu", weight_decay=0.002),
        DenseLayer(12, 12, "swish", weight_decay=0.002),
        DenseLayer(12, 3, "id", weight_decay=0.002)
    ],
    loss_function=Huber(delta=1.0), # for regression tasks
    optimizer=Momentum(learn_rate=0.05, momentum=0.75)
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
Evaluate the accuracy of the network:
```python
nn.metrics.check_accuracy(
	test_input=input_test_list,
	test_output=output_test_list
)
```
*Note: only regression, multilabel and multiclass accuracy calculations are supported as of now.*

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
        DenseLayer  (4, 10, "tanh",                      weight_decay=0.001),
        DropoutLayer(10, 16, "tanh",   dropout_rate=0.1, weight_decay=0.001),
        DenseLayer  (16, 12, "tanh",                     weight_decay=0.001),
        DenseLayer  (12, 3, "sigmoid",                   weight_decay=0.001)
    ],
    loss_function=BCE(),
    optimizer=Momentum(learn_rate=0.04, momentum=0.75)
)
```
```
Detected sigmoid in the last layer. Running accuracy check for multilabel.
Accuracy on 20,000 samples
Accuracy per output:    94.77%   90.57%   94.07%
                   Expected |                   Predicted | Input Data
   1.0000   1.0000   0.0000 |    0.9335   0.9909   0.0357 |  -4.835 -4.507 -4.715  3.635
   1.0000   0.0000   1.0000 |    0.9740   0.5168   0.9875 |   1.751  4.403  0.534  3.549
   1.0000   0.0000   0.0000 |    0.9946   0.2209   0.0125 |  -2.063  5.013 -3.300 -4.269
   0.0000   0.0000   1.0000 |    0.0019   0.0323   0.9758 |  -1.672 -5.604  2.177 -2.122
   0.0000   0.0000   0.0000 |    0.2262   0.4263   0.0165 |   2.239 -0.714 -0.690 -1.505
   1.0000   0.0000   1.0000 |    0.8852   0.0586   0.9939 |   2.209  0.099  4.055  5.609
   1.0000   0.0000   0.0000 |    0.9932   0.1647   0.0207 |   2.427  4.934 -2.039 -5.523
   0.0000   0.0000   1.0000 |    0.0020   0.0424   0.9730 |  -1.274 -3.315  2.458 -4.877
   1.0000   1.0000   0.0000 |    0.7447   0.9878   0.0155 |  -3.122 -4.311 -5.302 -1.944
   0.0000   0.0000   0.0000 |    0.0487   0.0358   0.3287 |   4.503 -1.770 -0.547  5.408
   1.0000   1.0000   0.0000 |    0.9840   0.9702   0.3160 |  -2.019  1.663 -5.119  2.274
   0.0000   0.0000   0.0000 |    0.3265   0.0161   0.0459 |   1.728  4.521 -4.578  4.024
   0.0000   1.0000   0.0000 |    0.0173   0.9692   0.0094 |  -0.369 -3.473 -0.763 -3.852
   1.0000   1.0000   1.0000 |    0.3761   0.8373   0.9270 |  -0.331  0.228 -5.740  3.879
   0.0000   1.0000   0.0000 |    0.0850   0.9794   0.0090 |  -2.292 -3.927 -3.937 -2.371
   0.0000   1.0000   0.0000 |    0.1021   0.9454   0.0516 |   0.332 -0.431 -2.518 -0.596
```

### Multiclass Classification Performance
**Multiclass classification** is where each input belongs to exactly one class. This model uses **Softmax** activation in the output layer and **categorial cross-entropy (CCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        DropoutLayer(4, 12, "prelu",   dropout_rate=0.2,                  weight_decay=0.001),
        DropoutLayer(12, 16, "tanh",   dropout_rate=0.2, batch_wise=True, weight_decay=0.001),
        DropoutLayer(16, 12, "swish",  dropout_rate=0.2,                  weight_decay=0.001),
        DenseLayer  (12, 3, "softmax",                                    weight_decay=0.001)
    ],
    loss_function=CCE(),
    optimizer=Momentum(learn_rate=0.02, momentum=0.75)
)
```
```
Detected softmax in the last layer. Running accuracy check for multiclass.
Accuracy on 20,000 samples
Accuracy per output:    95.40%   91.58%   96.43%
Overall categorization accuracy:    93.60%
                   Expected |                   Predicted | Input Data
   1.0000   0.0000   0.0000 |    0.9801   0.0145   0.0053 |   4.382  3.212  0.448  3.892
   0.0000   1.0000   0.0000 |    0.0140   0.9798   0.0062 |   1.039  0.098  5.527  2.458
   0.0000   0.0000   1.0000 |    0.0164   0.0028   0.9809 |   0.505  4.559  3.416  0.832
   0.0000   0.0000   1.0000 |    0.0124   0.0164   0.9712 |   1.724  3.400  4.717  0.230
   1.0000   0.0000   0.0000 |    0.6348   0.3513   0.0139 |   3.543  2.329  3.479  3.229
   0.0000   0.0000   1.0000 |    0.0409   0.0059   0.9532 |   1.162  2.467  1.510  1.547
   1.0000   0.0000   0.0000 |    0.0443   0.9513   0.0044 |   2.589  1.146  4.011  3.792
   0.0000   1.0000   0.0000 |    0.0145   0.9794   0.0061 |   3.661  1.245  5.298  1.416
   1.0000   0.0000   0.0000 |    0.9606   0.0372   0.0021 |   4.279  1.430  0.482  3.278
   1.0000   0.0000   0.0000 |    0.9636   0.0259   0.0105 |   2.797  3.261  2.216  3.402
   1.0000   0.0000   0.0000 |    0.9771   0.0143   0.0086 |   2.254  4.153  1.179  3.902
   1.0000   0.0000   0.0000 |    0.4998   0.4945   0.0058 |   2.378  1.985  3.166  4.179
   0.0000   0.0000   1.0000 |    0.0167   0.0033   0.9800 |   1.296  3.852  0.821  0.993
   0.0000   1.0000   0.0000 |    0.0762   0.9119   0.0119 |   2.179  1.617  3.127  2.739
   1.0000   0.0000   0.0000 |    0.9464   0.0506   0.0030 |   4.725  2.371  3.078  3.714
   1.0000   0.0000   0.0000 |    0.9575   0.0403   0.0021 |   4.396  1.340  1.510  3.959
```

### Regression Performance
**Regression** is where each input is mapped to a continuous value, rather than a discrete class. This model uses **Linear (Identity)** activation in the output layer and **mean-squared error (MSE)** loss for training.

*Note: Regression tasks generally require more rigorous data preparation and training control. Since output values can have a wider range compared to classification tasks, achieving high performance often demands more careful tuning of the model and training process.*
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        DenseLayer  (4, 12, "prelu",                                     weight_decay=0.001),
        DropoutLayer(12, 16, "tanh",  dropout_rate=0.1, batch_wise=True, weight_decay=0.001),
        DropoutLayer(16, 12, "swish", dropout_rate=0.1,                  weight_decay=0.001),
        DenseLayer  (12, 3, "id",                                        weight_decay=0.001)
    ],
    loss_function=Huber(delta=1.0),
    optimizer=Adam(learn_rate=0.01)
)
```
```
Detected id in the last layer. Running accuracy check for regression.
Mean Squared Error on 20,000 samples
Mean Squared Error per output:    10.49    8.62    6.45
                   Expected |                   Predicted | Input Data
   3.3575   8.6284   2.0535 |    4.3044   7.0457   3.2634 |  -0.477  1.323  1.722 -1.149
   2.7508  12.1666  10.5097 |    2.9617   9.1734   9.1678 |  -1.294  1.482  2.548  2.170
   6.6238  -6.0965   0.6740 |    5.4853  -4.6592   1.2948 |  -0.567  0.732 -1.052  1.659
  -3.4430  -2.5150   3.8812 |   -1.1821  -3.3759   2.5114 |  -0.526 -0.603 -0.282  1.821
 -13.8840   5.9988  -6.5019 |  -14.2842   6.2277  -3.9771 |   1.020 -2.153  0.174 -2.594
 -17.6580  -4.4838  -2.7950 |  -12.7972  -6.4664  -5.0047 |  -2.922 -0.688  2.721  0.720
  10.7379  10.2664   4.8267 |    7.7191   4.6950   3.5262 |  -0.108  1.782  0.897 -0.521
   8.4002  -4.2949   2.0537 |    6.5684  -4.1226   0.9138 |  -0.616  1.107 -0.611  1.614
 -17.9570  11.0277   4.5498 |  -16.9253   8.7957   3.9602 |   1.865 -3.061 -0.416 -0.702
   6.6330   6.4195  -2.6285 |    5.2088   8.2387   2.6677 |   2.963 -2.033  1.470  2.273
  -4.6607   1.4500  -2.7037 |   -3.5027   2.3028  -4.0241 |  -2.310  0.734  2.595 -2.845
  -5.2267   2.6799  -1.7481 |  -10.8543   4.6627  -1.7596 |   1.865  0.004 -1.047 -1.951
  -9.9641  -9.2251  -1.0424 |   -8.9206  -6.6737  -1.5643 |  -1.671 -1.222  0.538  1.052
  -6.8193  -2.7790  -1.0872 |   -6.8343  -4.9253  -1.0165 |  -2.280  0.075  1.504 -0.221
  -2.8018  10.5346   1.8449 |    0.8403   9.9028   1.7210 |   2.909  0.152 -0.170 -0.496
 -17.1167   9.0309   9.3532 |  -15.1465   7.5856   8.9784 |   2.454 -1.220 -2.901  0.861
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
- Different Optimizers options!?

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

$$Loss_{MSE}=( y_{actual} -y_{predicted})^{2}$$

### Binary Cross-Entropy (BCE)
When **MSE** struggles with small error values (less than 1) due to squaring, which diminishes those values, **Binary Cross-Entropy** is the answer. The logarithmic functions help amplify the error between 0 and 1, making it much more effective at penalizing incorrect predictions in this tiny range. This gives **BCE** an edge, especially in binary classification tasks where the model's predictions need to be confident (either 0 or 1).

$$Loss_{BCE}=-y_{actual}\log(y_{predicted})-(1-y_{actual})\log(1-y_{predicted})$$

### Categorial Cross-Entropy (CCE)
While **BCE** penalizes both 1s for not being 0 and 0s for not being 1, **Categorical Cross-Entropy** (CCE) focuses only on punishing the 0s when they should be 1. Designed to work with one-hot encoded values *together* with the **Softmax** activation function, **CCE** can guide a neural network to categorize data accurately by heavily penalizing incorrect 0 predictions. Once low raw scores become high enough, the **Softmax** activation will convert them into a high probability distribution.

$$Loss_{CCE}=-y_{actual}\log(y_{predicted})$$

## Optimization Techniques

### L2 (Ridge) Regularization
Just like in life, being an extremist is never good. The same goes for neural networks. Both underfitting and overfitting are problems to avoid:

- **Underfitting** happens when a network isn't able to learn well enough. This can happen due to various factors like a low neuron count, lack of non-linearity, or too low of a learning rate.

*Monkey Language: The network can't learn because it does not have enough brain cells or its brain cells are too simple.*
- **Overfitting** occurs when the network performs too perfectly on the training data, but fails to predict unseen data well. Essentially, the network "memorizes" the data instead of learning the underlying pattern. This usually happens due to overtraining, overwhelming number of parameters, overly strong gradients, or small datasets, etc.

*Monkey Language: The network memorized the material taught in class without understanding it, only to fail the exam when unseen questions come up.*

Rather than trying to balance everything from the start, we begin with a highly capable neural network. Possibly, multiple layers and appropriate activation functions. Then, we introduce a bit of interference in the learning process to prevent the network from learning too perfectly. This is exactly what **L2 (Ridge) Regularization** does.

$$Regularized\ Loss=Loss+\frac{1}{2} \lambda w^{2}$$

What is the intuition? In neural networks, large parameter values have a stronger influence on the output. If these large parameters begin to "memorize" the given data, the **L2 regularization** term (weight decay), $\frac{1}{2} \lambda w^{2}$ adds to the overall loss and heavily penalizes them with the squared values. This makes stronger parameters decay more quickly back toward zero. In other words, it is like a void pulling all parameters toward zero, ensuring that no value in the network explodes into huge negatives or positives. This process thereby reduces the risk of "memorizing" the data and helps the network generalize better. The $\lambda$ controls the strength of regularization.

### Momentum
In physics, when an object is moving, it has momentum. Objects with momentum continue to maintain that momentum unless external forces act upon them. Similarly, in neural networks, when parameters are moving toward a local minimum to minimize the loss, the momentum technique gives them the ability to "glide." This "gliding" helps them escape high-loss plateaus faster, allowing them to reach the local minimum more efficiently. It introduces the concept of "velocity" for each parameter in the model. The velocity is updated as follows:

$$v_{t+1} =\beta v_{t} +( 1-\beta ) \nabla L( w_{t})$$

where $\beta$ is the momentum strength, ranging between 0 and 1. A value of 0 disables the "gliding" behavior while 1 results in full "gliding" with no update. Ideally, $\beta$ is set between 0.5 and 0.8.

And the calculated velocity is applied to parameters as follows:

$$w_{t+1} =w_{t} -\alpha v_{t+1}$$

Here, $\alpha$ is simply the **learn rate**, which controls how fast or slow the velocity moves the parameter. Ideally, $\alpha$ values less than 0.01 are preferred. It can go as low as 0.0001 in some cases.