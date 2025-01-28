
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
- [GPU Acceleration?](#gpu-acceleration)
- [Resources *& Inspirations :)*](#resources--inspirations-)

## Features
- **Architecture**: Fully-connected layers with customizable layer sizes.
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
# Example 1: A network with 4 input neurons, 6 hidden neurons, and 2 output neurons
nn = NeuralNetwork([4, 6, 2])

# Example 2: A deeper network with multiple hidden layers
nn = NeuralNetwork([4, 8, 6, 6, 2])
```
#### Parameters
* `activation`: Activation functions for individual layers (E.g., `"relu"`, `"sigmoid"`).
* `loss_function`: Loss function for training (E.g., `"MSE"`, `"BCE"`)
* `learn_rate`: Learning rate for gradient descent.
* `lambda_parem`: Regularization strength to prevent overfitting.
* `momentum`: Momentum to accelerate convergence.

Example with added parameters:
```python
nn = NeuralNetwork(
    layers=[4, 12, 12, 3],
    activation=["tanh", "tanh", "sigmoid"], # activation for individual layers 
    loss_function="BCE", # for multilabel classification
    learn_rate=0.08,
    lambda_parem=0.003,
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
    layers=[4, 12, 12, 3],
    activation=["prelu", "prelu", "sigmoid"],
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
    layers=[4, 12, 12, 3],
    activation=["swish", "swish", "softmax"],
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
    layers=[4, 12, 12, 3],
    activation=["prelu", "prelu", "id"],
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

# Abyssal Zone
You made it this far. A scrolling enthusiast. Let's see where this takes you.

## Activation Functions Implemented
### Sigmoid
An 'S' shaped function very commonly used in neural networks. Since output ranges between 0 and 1, sigmoid is useful for classification tasks.
```math
A(x)=\frac{1}{1+e^{-x}}
```

$$A(x)=\frac{1}{1+e^{-x}}$$