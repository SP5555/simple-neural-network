# Simple Neural Network

## Overview
This project started out as a simple Python experiment to explore neural networks. But, well, things escalated. Now, it's a fully functional and **modular** neural network built completely from scratch.

While it is not exactly "simple" anymore, it is still a fun and *powerful* example of how a pile of numbers can learn to predict (*better than humans*).

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
### Architecture
- Feedforward Neural Network
### Layers
- **Dense**: Fully-connected layer
- **Dropout**: Randomly drops neurons during training to reduce overfitting.
- **BatchNorm**: Normalizes activations within a batch to stabilize and accelerate training.
### Activation functions
- **Bounded**: Sigmoid, Softmax, Tanh
- **Unbounded**: Linear, ReLU, Leaky ReLU, PReLU (Learnable), Softplus, Swish (Fixed/Learnable)
### Loss functions
- **Regression**: Mean-Squared Error (MSE), Mean-Absolute Error (MAE), Huber Loss
- **Multilabel classification**: Binary Cross-Entropy (BCE)
- **Multiclass classification**: Categorial Cross-Entropy (CCE)
### Training Algorithms
- Mini-batch gradient descent
### Optimizers
- Adam, AdaGrad, Momentum, RMSprop, Stochastic Gradient Descent (SGD)
### Advanced Features
- **Automatic Differentiation** ([LAZY Differentiation](https://github.com/SP5555/lazy-differentiation/)): A custom-built auto-differentiation engine that abstracts away the heavy math load (with NumPy at its core) for efficient backpropagation.
- **Tensor-based Computation Flow**: Similar to modern **PyTorch** and **TensorFlow** frameworks, this project uses custom-built tensors as core computation units, powered by auto-differentiation engine mentioned above.

## Dependencies
- **Python 3.11** or **3.12**
- **NumPy**

## Installation Instructions
Run the following commands to install it locally. Require `pip`.
- Clone the repository
```
git clone --recurse-submodules https://github.com/SP5555/simple-neural-network.git
```
- Change to the project directory
```
cd simple-neural-network
```
- OPTIONAL if submodules are not working
```
# git submodule update --init --recursive
```
- Create a virtual environment
```
python -m venv venv
```
- Activate the virtual environment
```
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
- Install the required dependencies
```
pip install numpy
# numpy is enough, don't install -r requirements.txt
```

## Usage

### Creating a Neural Network
To create a neural network with customizable layer configurations:

**Parameters**
* `layers`: List of supported layer classes.
* `loss_function`: Loss function for training (E.g., `MSE`, `BCE`)
* `optimizer`: an instance of a derived optimizer class (E.g., `SGD`, `Momentum`)

**Example 1**: A network with 4 input neurons, 6 hidden neurons, and 2 output neurons.
Leaky Relu activation in hidden layer and sigmoid activation in final layer, SGD optimizer and BCE Loss function
```python
nn = NeuralNetwork(layers=[
        Dense(6, activation=LeakyReLU()),
        Dense(2, activation=Sigmoid()),
    ],
    loss_function=BCE(),
    optimizer=SGD(learn_rate=0.02)
)
nn.build(input_size=4)
```
**Example 2**: added decaying rates and dropout layer
```python
nn = NeuralNetwork(
    layers=[
        Dense(12, activation=PReLU(),  weight_decay=0.002),
        Dense(12, activation=Swish(),  weight_decay=0.002),
        Dropout(dropout_rate=0.2),
        Dense(3,  activation=Linear(), weight_decay=0.002)
    ],
    loss_function=Huber(delta=1.0), # for regression tasks
    optimizer=Momentum(learn_rate=0.05, momentum=0.75)
)
nn.build(input_size=4)
```
*Note: `build(input_size)` must be called **before** training or inference to initialize layers and compile the computation graph.*


### Training
To train a network with input and output data:
```python
nn.train(
	input_list=input_train_list,
	output_list=output_train_list,
	epoch=1000,
	batch_size=16 # number of samples in each mini-batch for training
)
```
### Utilities

~~View the current weights and biases of the network:~~

**This function needs some rework. It might run into errors.**

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
        Dense(10, activation=Tanh(),    weight_decay=0.001),
        Dense(16, activation=Tanh(),    weight_decay=0.001),
        Dropout(dropout_rate=0.4),
        Dense(12, activation=Tanh(),    weight_decay=0.001),
        Dense(3,  activation=Sigmoid(), weight_decay=0.001)
    ],
    loss_function=BCE(),
    optimizer=Momentum(learn_rate=0.04, momentum=0.75)
)
nn.build(input_size=4)
```
```
Detected Sigmoid in the last layer. Running accuracy check for multilabel.
Accuracy on 20,000 samples
Accuracy per output:    91.20%   91.09%   93.57%
          Expected |          Predicted | Input Data
  0.00  0.00  1.00 |   0.01  0.28  0.81 |   2.38 -0.00 -2.95  0.73
  0.00  1.00  1.00 |   0.70  0.94  0.98 |  -3.00  3.82  3.97 -5.05
  1.00  1.00  1.00 |   0.87  0.98  0.41 |  -2.82 -0.08 -4.99 -0.71
  0.00  0.00  1.00 |   0.01  0.13  0.97 |  -1.97 -4.38  1.59  5.41
  0.00  1.00  1.00 |   0.35  0.98  0.97 |  -5.20  4.98  4.66  0.55
  0.00  1.00  1.00 |   0.72  0.95  0.97 |  -2.91  5.77  4.59 -3.67
  0.00  0.00  1.00 |   0.01  0.12  0.95 |  -3.01 -2.03  3.16 -2.76
  0.00  0.00  1.00 |   0.02  0.08  0.98 |  -4.50 -5.34  4.79  4.15
  1.00  0.00  0.00 |   0.96  0.09  0.00 |   2.64  6.10 -2.39  4.32
  0.00  1.00  1.00 |   0.34  0.98  0.97 |  -4.54  4.06  4.56 -1.43
  0.00  1.00  1.00 |   0.32  0.98  0.92 |  -4.74 -0.50  0.68  0.95
  1.00  1.00  1.00 |   0.98  0.88  0.23 |  -2.00  5.34 -0.03  3.54
  1.00  0.00  0.00 |   0.98  0.14  0.00 |  -3.27  5.76 -4.50 -1.82
  1.00  0.00  0.00 |   0.95  0.18  0.71 |   2.53  2.36  4.29 -5.28
  1.00  1.00  0.00 |   0.97  0.96  0.23 |  -3.91  1.68 -1.79  4.45
  0.00  0.00  1.00 |   0.13  0.03  0.99 |   2.87 -5.87  4.61  3.88
```

### Multiclass Classification Performance
**Multiclass classification** is where each input belongs to exactly one class. This model uses **Softmax** activation in the output layer and **categorial cross-entropy (CCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        Dense(12, activation=PReLU(),   weight_decay=0.001),
        Dense(16, activation=Tanh(),    weight_decay=0.001),
        Dropout(dropout_rate=0.4, batch_wise=True),
        Dense(12, activation=Swish(),   weight_decay=0.001),
        Dropout(dropout_rate=0.4),
        Dense(3,  activation=Softmax(), weight_decay=0.001)
    ],
    loss_function=CCE(),
    optimizer=Adam(learn_rate=0.02)
)
nn.build(input_size=6)
```
```
Detected Softmax in the last layer. Running accuracy check for multiclass.
Accuracy on 20,000 samples
Accuracy per output:    95.06%   91.16%   92.04%
Overall categorization accuracy:    92.03%
          Expected |          Predicted | Input Data
  0.00  0.00  1.00 |   0.70  0.02  0.28 |   3.04  1.43  0.34  0.08 -0.90 -0.42
  0.00  1.00  0.00 |   0.00  0.99  0.01 |   0.52 -0.46  5.91  3.83  0.82 -3.21
  0.00  0.00  1.00 |   0.01  0.01  0.98 |   1.41  0.54  4.12 -0.17  2.32 -2.57
  0.00  0.00  1.00 |   0.01  0.00  0.99 |   2.25  0.70  1.20 -1.19  0.68 -2.62
  0.00  0.00  1.00 |   0.00  0.00  1.00 |   2.21  0.85  0.20 -3.40  1.36 -1.72
  0.00  0.00  1.00 |   0.00  0.00  1.00 |   2.43 -0.90  2.41  0.97 -0.59  2.71
  0.00  1.00  0.00 |   0.01  0.82  0.17 |  -0.15 -1.36  2.31  1.87  1.43 -2.10
  0.00  1.00  0.00 |   0.01  0.40  0.59 |   1.53 -2.08  4.02  1.56  0.66 -0.22
  0.00  1.00  0.00 |   0.30  0.64  0.06 |   3.50  1.01  1.33  2.18  1.96  0.68
  1.00  0.00  0.00 |   0.99  0.00  0.01 |   0.19  3.21  0.91  0.86 -1.84  1.68
  1.00  0.00  0.00 |   0.19  0.01  0.80 |   0.71  1.21  3.87 -1.40  0.90  1.30
  1.00  0.00  0.00 |   1.00  0.00  0.00 |  -1.02  3.94  2.98  1.77 -1.66 -1.67
  0.00  0.00  1.00 |   0.00  0.00  1.00 |   2.89  1.67 -1.47 -1.08  1.21 -2.23
  0.00  0.00  1.00 |   0.00  0.00  1.00 |   1.88 -0.40 -1.96  1.03 -1.31 -0.79
  1.00  0.00  0.00 |   0.99  0.01  0.00 |   0.74  4.68  2.27  3.44  1.10 -1.03
  0.00  1.00  0.00 |   0.01  0.98  0.00 |   3.05  2.24  1.36  3.60  3.04 -4.69
```

### Regression Performance
**Regression** is where each input is mapped to a continuous value, rather than a discrete class. This model uses **Linear (Identity)** activation in the output layer and **Huber** loss for training.

*Note: Regression tasks generally require more rigorous data preparation and training control. Since output values can have a wider range compared to classification tasks, achieving high performance often demands more careful tuning of the model and training process.*
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        Dense(12, activation=PReLU(),  weight_decay=0.001),
        Dense(16, activation=Tanh(),   weight_decay=0.001),
        Dropout(dropout_rate=0.4),
        Dense(12, activation=Swish(),  weight_decay=0.001),
        Dropout(dropout_rate=0.4),
        Dense(3,  activation=Linear(), weight_decay=0.001)
    ],
    loss_function=Huber(delta=1.0),
    optimizer=Adam(learn_rate=0.01)
)
nn.build(input_size=4)
```
```
Detected Linear in the last layer. Running accuracy check for regression.
Mean Squared Error on 20,000 samples
Mean Squared Error per output:    19.04   13.44   11.15
          Expected |          Predicted | Input Data
  4.25 -7.45  1.60 |   3.76 -6.07 -0.61 |  -0.41  0.51 -1.59  1.50
-12.06 -2.77 -2.61 |  -8.52 -0.64 -4.84 |   0.14 -2.69  0.87 -0.00
 15.87-15.17 -7.63 |   9.25 -9.91 -3.47 |  -1.14  2.17 -0.89 -0.22
 18.22  4.46  5.15 |  11.65  3.30  2.53 |   1.20  3.26 -0.61  2.35
 14.30-18.07 -5.33 |  11.98-15.76 -5.73 |  -2.15  1.91 -2.22  0.94
  5.68 -0.11 -0.52 |   5.23 -1.24 -0.21 |   0.55  1.64 -0.52  0.00
 -1.62 -5.27  4.10 |  -2.07 -2.02  1.21 |  -0.69 -0.94 -2.09  0.20
  9.10-16.90  1.01 |   5.79-10.57 -1.79 |  -2.42  0.26 -2.21  2.93
  3.13 -0.77  2.80 |   1.70  0.31  2.13 |   0.44  0.24 -0.53  1.75
-10.35  4.14 10.78 |  -8.42  4.41  6.48 |   0.85 -1.91 -2.07  1.77
 17.24-20.91-12.06 |  10.50-16.79 -7.19 |  -1.39  2.44 -2.03 -1.21
 -8.25 -1.68 14.72 |  -8.15  2.05  6.79 |  -1.00 -2.90 -1.80  2.77
  8.87 11.83  3.79 |   9.57 10.02  4.66 |   1.90 -0.22  1.72  1.41
 -5.83  0.29 -0.77 |  -3.12  0.24 -0.25 |   0.72 -0.09 -1.52 -0.71
-12.48  3.05  4.13 |  -9.91  4.27  3.99 |  -0.65 -3.05 -1.95 -1.19
-14.17-10.39 -3.07 | -10.46 -5.07 -4.39 |  -1.82 -2.70  0.47  0.68
```
As shown, the neural network performs exceptionally well on the synthetic data. If real-world data exhibits similar relationships between inputs and outputs, the network is likely to perform equally well.

> *Well, real-world data is insanely messy. It's full of noise, outliers, and all sorts of headaches. But here's the debate: who's responsible for dealing with it? Should the programmer clean up the data before feeding it to the network, or should the neural network learn to handle the chaos on its own?*

## Synthetic Data Generation
With randomization, we can create diverse datasets that still follow a certain input-output relationship pattern to some extent (just like in the real world). This mimics the real world data and helps test the network's ability to generalize and perform well across different situations.

To note, we are NOT harvesting the full power of randomization as it would only generate complete gibberish. Instead, we establish a relation between inputs and outputs while introducing randomness to it at the same time. This synthetically generated data helps in evaluating the model's performance without the need for a real-world dataset, which can be difficult to acquire or pre-process.

> *Monkey Language: The idea is, we define input-output relationships and generate data while heavily masking them with a reasonable level of randomness, making the patterns not immediately clear, even to humans. Then we evaluate the network's ability to cut through those random noise and uncover the underlying pattern.*

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

This neural network is designed with **modularity and extensibility** in mind, which means only one thing. YOU, whoever you are, can add custom layers, activations, or whatever components in here easily as you see fit.

Feel free to mess around with the hyperparameters, or change things up entirely. You might discover some cool insights or get the network to do things it wasn't originally designed for. *What is it originally designed for anyway?*

*But hey, no one gets hurt if this thing breaks.*

## What to Expect?
This network has **insane level of scalibility** in terms of depth (number of layers) and width (neurons per layer), with the only real limit being your hardware, super impressive! Dense network architectures are suited the most for general-purpose tasks since they can approximate *any* mathematical function.

> *Monkey Language: We're saying that stacks of linear equations and activation functions can approximate even the most complicated mathematical relations to some extent, similar to how taylor series approximation works. As long as there's an underlying mathematical pattern between the input and output, this network can learn to model it!*

Meanwhile, due to the **Lack of Structure Awareness**, the input is just a big flat vector of numbers to this network. And therefore, it can NOT understand the spatial hierarchies like **Convolutional Neural Networks (CNN)** and sequential dependencies like **Recurrent Neural Networks (RNN)**.

> *Monkey Language: this means, you can't train this dense network to recognize images, or "understand" the meaning of texts and time-series data efficiently.*

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
- Chill out. Not today.

# Abyssal Zone (Recommended to not proceed)
You made it this far. A scrolling enthusiast. Let's see where this takes you.

- [Activation Functions](neural_network/activations/README.md)
- [Loss Functions](neural_network/losses/README.md)
- [Optimizers](neural_network/optimizers/README.md)

## Regularization Techniques

### L2 (Ridge) Regularization
Just like in life, being an extremist is never good. The same goes for neural networks. Both underfitting and overfitting are problems to avoid:

- **Underfitting** happens when a network isn't able to learn well enough. This can happen due to various factors like a low neuron count, lack of non-linearity, or too low of a learning rate.

> *Monkey Language: The network can't learn because it does not have enough brain cells or its brain cells are too simple.*
- **Overfitting** occurs when the network performs too perfectly on the training data, but fails to predict unseen data well. Essentially, the network "memorizes" the data instead of learning the underlying pattern. This usually happens due to overtraining, overwhelming number of parameters, overly strong gradients, or small datasets, etc.

> *Monkey Language: The network memorized the material taught in class without understanding it, only to fail the exam when unseen questions come up.*

Rather than trying to balance everything from the start, we begin with a highly capable neural network. Possibly, multiple layers and appropriate activation functions. Then, we introduce a bit of interference in the learning process to prevent the network from learning too perfectly. This is exactly what **L2 (Ridge) Regularization** does.

$$Regularized\ Loss=Loss+\frac{1}{2} \lambda w^{2}$$

What is the intuition? In neural networks, large parameter values have a stronger influence on the output. If these large parameters begin to "memorize" the given data, the **L2 regularization** term (weight decay), $\frac{1}{2} \lambda w^{2}$ adds to the overall loss and heavily penalizes them with the squared values. This makes stronger parameters decay more quickly back toward zero. In other words, it is like a void pulling all parameters toward zero, ensuring that no value in the network explodes into huge negatives or positives. This process thereby reduces the risk of "memorizing" the data and helps the network generalize better. The $\lambda$ controls the strength of regularization.
