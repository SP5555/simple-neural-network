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

**Example 1**: A network with 4 input neurons, 6 hidden neurons, and 2 output neurons. Uses **Leaky ReLU** activation in the hidden layer and **Sigmoid** activation in the final layer.
```python
nn = NeuralNetwork(
    layers=[
        Dense(6, activation=LeakyReLU()),
        Dense(2, activation=Sigmoid()),
    ]
)
nn.build(input_size=4)
```

**Example 2**: added decaying rates and a **Dropout** layer.
```python
nn = NeuralNetwork(
    layers=[
        Dense(12, activation=PReLU()),
        Dense(12, activation=Swish()),
        Dropout(dropout_rate=0.2),
        # declaring weight for individual layers
        # overrides the global weight decay
        Dense(3,  activation=Linear(), weight_decay=0.001)
    ],
    weight_decay=0.002 # global weight decay
)
nn.build(input_size=4)
```
*Note: `build(input_size)` must be called before training or inference to initialize layers and compile the computation graph.*



### Training
Create a trainer instance:

**parameters**
* `model`: an instance of `NeuralNetwork`
* `loss_function`: Loss function for training (E.g., `MSE`, `BCE`)
* `optimizer`: an instance of a derived optimizer class (E.g., `SGD`, `Momentum`)

**Example 1**: trainer with **SGD** optimizer and **BCE** Loss function.
```python
trainer = Trainer(
    nn,
    loss_function=BCE(),
    optimizer=SGD(learn_rate=0.02)
)
```
**Example 2**: for regression tasks with **Momentum** optimizer and **Huber** Loss function.
```python
trainer = Trainer(
    nn,
    loss_function=Huber(delta=1.0), # for regression tasks
    optimizer=Momentum(learn_rate=0.05, momentum=0.75)
)
```
*The `Trainer` will automatically connect the loss function to the model's output and handle gradient propagation during training.*


To train a network with input and output data:
```python
trainer.train(
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

The **synthetic** data (artificial data created using algorithms) is used to test the model's ability to predict outcomes based on the input features. All tests were conducted with **2,000 training samples**. The performance is then evaluated on **1,000 unseen test samples** generated using the same method (*the predictions for 16 unseen samples are compared with the expected outputs below*). [How is synthetic data generated?](#synthetic-data-generation)

### Multilabel Classification Performance
**Multilabel classification** is where each input can belong to multiple classes simultaneously. This model uses **Sigmoid** activation in the output layer and **binary cross-entropy (BCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        Dense(10, activation=Tanh()),
        Dense(16, activation=Tanh()),
        Dropout(dropout_rate=0.4),
        Dense(12, activation=Tanh()),
        Dense(3,  activation=Sigmoid())
    ],
    weight_decay=0.001
)
nn.build(input_size=4)
trainer = Trainer(
    nn,
    loss_function=BCE(),
    optimizer=Adam(learn_rate=0.02)
)
```
```
Detected Sigmoid in the last layer. Running accuracy check for multilabel.
Multilabel Metrics on 1,000 samples  (threshold = 0.5)
             Precision    Recall        F1
  Label  1       0.9571    0.9494    0.9533
  Label  2       0.9078    0.8917    0.8997
  Label  3       0.9355    0.9591    0.9472
  Exact Match (all labels correct): 80.60%
          Expected |          Predicted | Input Data
  0.00  1.00  0.00 |   0.00  0.95  0.02 |  -1.04 -5.49 -0.77  4.64
  0.00  1.00  0.00 |   0.00  0.45  0.06 |  -6.00  0.11  5.22 -3.83
  1.00  0.00  0.00 |   1.00  0.08  0.05 |   1.30  3.50 -0.18 -1.99
  1.00  1.00  1.00 |   0.78  0.99  0.83 |  -3.01  1.07  0.13 -1.46
  1.00  1.00  1.00 |   0.99  0.85  0.99 |   1.92  4.83  1.17 -3.89
  0.00  0.00  1.00 |   0.02  0.02  0.96 |   3.34 -3.34  1.12  2.67
  1.00  0.00  1.00 |   0.89  0.00  0.95 |   3.93 -2.78  0.56 -2.24
  0.00  1.00  0.00 |   0.00  0.96  0.02 |  -1.67 -4.89 -1.40 -6.03
  1.00  0.00  0.00 |   0.97  0.05  0.05 |   2.89  5.74 -0.57  0.31
  0.00  0.00  1.00 |   0.01  0.02  0.96 |   1.35 -3.88  0.47  0.41
  1.00  1.00  0.00 |   0.60  0.98  0.01 |  -3.09 -0.92 -2.14 -1.95
  1.00  1.00  0.00 |   1.00  0.76  0.01 |  -4.76  3.97 -0.72  4.91
  0.00  0.00  0.00 |   0.04  0.03  0.03 |   3.77  1.14 -3.81 -3.24
  0.00  0.00  1.00 |   0.01  0.04  0.95 |  -3.08 -3.08  2.14 -0.21
  1.00  0.00  0.00 |   0.81  0.02  0.04 |   5.26  1.63 -0.63 -1.23
  1.00  1.00  0.00 |   0.98  0.98  0.02 |  -5.56  5.10 -1.59 -4.03
```

### Multiclass Classification Performance
**Multiclass classification** is where each input belongs to exactly one class. This model uses **Softmax** activation in the output layer and **categorial cross-entropy (CCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        Dense(12, activation=PReLU()),
        Dense(16, activation=Tanh()),
        Dropout(dropout_rate=0.4, batch_wise=True),
        Dense(12, activation=Swish()),
        Dropout(dropout_rate=0.4),
        Dense(3,  activation=Softmax())
    ],
    weight_decay=0.001
)
nn.build(input_size=6)
trainer = Trainer(
    nn,
    loss_function=CCE(),
    optimizer=Adam(learn_rate=0.02)
)
```
```
Detected Softmax in the last layer. Running accuracy check for multiclass.
Multiclass Metrics on 1,000 samples
             Precision    Recall        F1
  Class  1       0.9164    0.9242    0.9203
  Class  2       0.9000    0.9623    0.9301
  Class  3       0.9635    0.8896    0.9250
  Overall Accuracy: 92.50%
          Expected |          Predicted | Input Data
  0.00  0.00  1.00 |   0.49  0.07  0.44 |   0.89  1.03  2.69  0.71 -1.27 -2.37
  0.00  0.00  1.00 |   0.00  0.00  1.00 |   2.01 -1.38 -0.98 -0.61 -0.42  3.65
  0.00  0.00  1.00 |   0.00  0.00  1.00 |   0.71 -1.70 -0.21 -0.35  2.51 -1.14
  0.00  0.00  1.00 |   0.00  0.00  1.00 |   0.58 -1.06  3.45 -3.48 -1.63  2.36
  0.00  1.00  0.00 |   0.96  0.04  0.01 |   0.78  2.77  1.80  2.14  1.45 -0.39
  0.00  1.00  0.00 |   0.00  1.00  0.00 |   3.14 -0.36  3.82  3.47  2.39 -4.11
  1.00  0.00  0.00 |   1.00  0.00  0.00 |  -1.24  4.53  1.19  1.73 -2.63  1.73
  0.00  1.00  0.00 |   0.00  1.00  0.00 |   0.06  0.43  4.17  2.75  3.58 -3.17
  1.00  0.00  0.00 |   0.96  0.02  0.02 |   2.34  2.80  2.60  1.18  1.10  0.90
  1.00  0.00  0.00 |   1.00  0.00  0.00 |   1.60  3.37  0.54  0.79 -2.76 -0.57
  1.00  0.00  0.00 |   0.99  0.01  0.01 |  -0.94  1.60  0.66  2.32 -0.77  1.18
  0.00  1.00  0.00 |   0.00  0.99  0.01 |  -0.11  2.88  2.00  1.76  3.76 -3.22
  1.00  0.00  0.00 |   1.00  0.00  0.00 |   0.50  3.68  2.83 -3.00 -2.97  1.46
  1.00  0.00  0.00 |   0.25  0.00  0.75 |  -0.46  1.30  3.97 -2.47  1.07  2.09
  1.00  0.00  0.00 |   0.86  0.14  0.00 |  -0.62  1.65  2.07  4.70 -2.32 -1.39
  1.00  0.00  0.00 |   0.95  0.01  0.04 |   1.05  1.89  3.55  2.18 -2.05  1.76
```

### Regression Performance
**Regression** is where each input is mapped to a continuous value, rather than a discrete class. This model uses **Linear (Identity)** activation in the output layer and **Huber** loss for training.

*Note: Regression tasks generally require more rigorous data preparation and training control. Since output values can have a wider range compared to classification tasks, achieving high performance often demands more careful tuning of the model and training process.*
```python
# Model configuration
nn = NeuralNetwork(
    layers=[
        Dense(12, activation=PReLU()),
        Dense(16, activation=Tanh()),
        BatchNorm(),
        Dropout(dropout_rate=0.4),
        Dense(12, activation=Swish()),
        Dense(3,  activation=Linear())
    ],
    weight_decay=0.001
)
nn.build(input_size=4)
trainer = Trainer(
    nn,
    loss_function=Huber(delta=2.5),
    optimizer=Adam(learn_rate=0.01)
)
```
```
Detected Linear in the last layer. Running accuracy check for regression.
Regression Metrics on 1,000 samples
              RMSE       MAE        R2
  Output 1   2.8721    2.2364    0.9325
  Output 2   3.0447    2.3375    0.9221
  Output 3   2.4687    1.9071    0.8858
          Expected |          Predicted | Input Data
 14.43  5.57  7.18 |  13.63  3.66  4.01 |  -2.37  1.40  1.57  0.91
 -3.33 -2.15  0.96 |  -2.88  0.38  2.34 |  -0.09 -1.02 -0.89  0.02
 -9.60  6.39  6.33 |  -6.77  5.36  6.15 |   0.58 -1.82 -2.08 -0.88
-13.97 11.64 -4.14 | -12.83  9.82 -2.50 |   2.09 -0.25  1.76 -2.17
 -9.17 -6.39-16.32 |  -5.85 -3.65-12.52 |  -0.61 -2.84  3.10 -0.22
 22.84  5.80  4.79 |  16.66 10.98  8.25 |   2.64  1.63 -0.40  2.30
  2.69 -8.21 -2.30 |   7.04 -6.58 -2.48 |  -2.49  0.53  0.81 -0.31
-13.66 -3.76  0.50 | -10.17 -3.71  1.16 |  -1.27 -1.50 -1.26 -1.28
 -4.50 -4.51  9.15 |  -4.93 -2.52  8.82 |  -0.34 -0.99 -2.04  3.00
-11.58-10.63  0.98 | -10.04 -9.76  1.49 |  -1.67 -0.48 -2.11  0.10
-10.35  4.14 -2.62 | -12.87  4.92 -4.28 |   1.26 -1.65  1.17 -0.89
  4.56 -9.70 -1.19 |   0.91-10.92 -2.45 |  -1.88  0.88 -0.69  0.96
 13.35 18.95 13.13 |  15.99 14.80 11.08 |   0.35  3.35  1.67  0.92
  4.70 -8.60-15.31 |   3.28 -4.76-10.32 |  -1.96 -1.25  2.25 -2.66
 -2.75  2.50 -0.92 |  -2.79  3.79  0.35 |   0.48 -0.62 -0.57 -1.15
 -5.31 14.17  4.49 |  -0.52 15.68  6.59 |   2.28  0.12  1.98  1.88
```
As shown, the neural network performs exceptionally well on the synthetic data. If real-world data exhibits similar relationships between inputs and outputs, the network is likely to perform equally well.

> *Well, real-world data is insanely messy. It's full of noise, outliers, and all sorts of headaches. But here's the debate: who's responsible for dealing with it? Should the programmer clean up the data before feeding it to the network, or should the neural network learn to handle the chaos on its own?*

## Synthetic Data Generation
With randomization, we can create diverse datasets that still follow a certain input-output relationship pattern to some extent (just like in the real world). This mimics the real world data and helps test the network's ability to generalize and perform well across different situations.

To note, we are NOT harvesting the full power of randomization as it would only generate complete gibberish. Instead, we establish a relation between inputs and outputs while introducing randomness to it at the same time. This synthetically generated data helps in evaluating the model's performance without the need for a real-world dataset, which can be difficult to acquire or pre-process.

> *Monkey Language: The idea is, we define input-output relationships and generate data while heavily masking them with a reasonable level of randomness, making the patterns not immediately clear, even to humans. Then we evaluate the network's ability to cut through those random noise and uncover the underlying pattern.*

For the above example runs, the data is generated as follows. Multilabel and regression use 4 input features (`i1`-`i4`) with 3 outputs; multiclass uses 6 input features with 3 one-hot class outputs.
```python
# For multilabel classification demonstration
rng = np.random.default_rng(seed)

i1 = rng.uniform(-6, 6, size=n)
i2 = rng.uniform(-6, 6, size=n)
i3 = rng.uniform(-6, 6, size=n)
i4 = rng.uniform(-6, 6, size=n)

# Define arbitrary relationships between inputs and outputs
o1 = (i1*i4 - 5*i2 < 2*i1*i3 - i4).astype(float)
o2 = (4*i1 - 2*i2*i3 + 0.4*i4*i2/i1 < -3*i3).astype(float)
o3 = (-i1/i4 + 0.3*i2 - 8*i2*i2/i3 < 2*i4).astype(float)

X = _add_noise(rng, np.column_stack((i1, i2, i3, i4)), noise=0.2)
Y = np.column_stack((o1, o2, o3))
```
```python
# For multiclass classification demonstration
rng = np.random.default_rng(seed)

n_inputs  = 6
n_classes = 3

X = np.zeros((n, n_inputs))
Y = np.zeros((n, n_classes))

class_labels = rng.integers(0, n_classes, size=n)

# Define input data ranges for each class
class_ranges = {
    0: [(-2, 3), (1, 5), (0, 4), (-3, 5), (-3, 3), (-2, 2)],
    1: [(-1, 4), (-2, 3), (1, 6), (1, 5),  (0, 6),  (-5, 1)],
    2: [(0, 3),  (-2, 2), (-2, 5), (-4, 2), (-2, 4), (-3, 5)],
}

for c in range(n_classes):
    idx = np.where(class_labels == c)[0]
    for feature, (low, high) in enumerate(class_ranges[c]):
        X[idx, feature] = rng.uniform(low, high, size=len(idx))
    Y[idx, c] = 1.0

X = _add_noise(rng, X, noise=0.2)
```
```python
# For regression demonstration
rng = np.random.default_rng(seed)

i1 = rng.uniform(-3, 3, size=n)
i2 = rng.uniform(-3, 3, size=n)
i3 = rng.uniform(-3, 3, size=n)
i4 = rng.uniform(-3, 3, size=n)

# Define arbitrary relationships between inputs and outputs
o1 = i1*i4 + 5*i2 - 2*i1*i3 + i4
o2 = 4*i1 + 2*i2*i3 + 0.4*i4*i2 + 3*i3
o3 = i1 + 0.3*i2 + 2*i3*i2 + 2*i4

X = _add_noise(rng, np.column_stack((i1, i2, i3, i4)), noise=0.5)
Y = np.column_stack((o1, o2, o3))
```
In all cases, input features are exposed to some noise to better mimic real-world scenarios.
```python
def _add_noise(rng: np.random.Generator, data: np.ndarray, noise: float) -> np.ndarray:
    return data + rng.uniform(-noise, noise, size=data.shape)
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
