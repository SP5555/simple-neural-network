
# Simple Neural Network

## Overview
This is the very simple, yet *powerful* example of a fully functional neural network implemented from scratch in Python (with numpy).

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
- **Fully connected layers**: Customizable layer sizes.
- **Activation functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, Linear Softmax.
- **Loss functions**: Mean-Squared Error (MSE) for regression, Binary Cross-Entropy (BCE) for multilabel classification, Categorial Cross-Entropy (CCE or MCE) for multiclass classification.
- **Training Algorithm**: Gradient descent.

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
	batch_size=64
)
```
### Utilities
View the current weights and biases of the network:
```python
nn.inspect_weights_and_biases()
```
Evaluate the accuracy of the network for binary classification:
```python
nn.check_accuracy_classification(
	test_input=input_test_list,
	test_output=output_test_list
)
```
Compare the predicted results with the desired results:
```python
nn.compare_predictions(input=data_i, output=data_o)
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
    activation=["tanh", "tanh", "sigmoid"],
    loss_function="BCE",
    learn_rate=0.08,
    lambda_parem=0.003,
    momentum=0.75
)
```
```
Accuracy on 20,000 samples
Accuracy on each output:    96.15%   93.52%   92.78%
                   Expected |                   Predicted | Input Data
   1.0000   1.0000   0.0000 |    0.9972   0.7317   0.0132 |  -1.328  2.723 -1.069 -2.377
   1.0000   0.0000   0.0000 |    0.9421   0.0040   0.1248 |   5.753  4.139 -0.224  1.315
   1.0000   1.0000   1.0000 |    0.9738   0.9990   0.8112 |  -3.634 -0.992 -4.692  4.433
   1.0000   1.0000   1.0000 |    0.9865   0.9940   0.4115 |  -4.126  1.910 -4.334  4.576
   0.0000   1.0000   0.0000 |    0.0994   0.9570   0.1016 |  -0.928 -3.851 -4.605  5.514
   0.0000   0.0000   1.0000 |    0.0086   0.4190   0.9559 |  -2.347 -5.100  0.653  4.223
   1.0000   1.0000   1.0000 |    0.5226   0.3747   0.9948 |  -0.260  0.773  1.272  1.559
   0.0000   0.0000   1.0000 |    0.0046   0.0360   0.9815 |   2.842 -3.701  0.761  3.926
   1.0000   0.0000   1.0000 |    0.9369   0.0052   0.5779 |   1.812 -1.078  5.590  1.029
   0.0000   0.0000   1.0000 |    0.0074   0.7400   0.9563 |  -4.343 -5.304  0.991  3.811
   1.0000   0.0000   1.0000 |    0.9981   0.1394   0.9961 |   4.879  3.402  4.317  4.496
   1.0000   0.0000   1.0000 |    0.9968   0.5184   0.9965 |   1.177  3.738  1.606  2.631
   0.0000   1.0000   0.0000 |    0.0088   0.9380   0.0429 |   0.347 -5.124 -5.869 -1.815
   0.0000   0.0000   1.0000 |    0.0057   0.2229   0.6551 |   1.970 -5.232  0.151  1.269
   1.0000   0.0000   0.0000 |    0.9937   0.0602   0.0091 |  -1.049  6.007 -5.412 -2.311
   0.0000   0.0000   0.0000 |    0.0250   0.8283   0.0390 |   0.519 -1.991 -0.483 -0.456
```

### Multiclass Classification Performance
**Multiclass classification** is where each input belongs to exactly one class. This model uses **Softmax** activation in the output layer and **categorial cross-entropy (CCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[4, 12, 12, 3],
    activation=["leaky_relu", "sigmoid", "softmax"],
    loss_function="CCE",
    learn_rate=0.02,
    lambda_parem=0.003,
    momentum=0.75
)
```
```
Accuracy on 20,000 samples
Accuracy on each output:    95.66%   93.80%   97.14%
Overall categorization accuracy:    93.34%
                   Expected |                   Predicted | Input Data
   1.0000   0.0000   0.0000 |    0.8853   0.1047   0.0100 |   2.392  1.273  1.295  4.643
   0.0000   1.0000   0.0000 |    0.0584   0.9257   0.0159 |   1.460  1.613  4.282  3.557
   0.0000   0.0000   1.0000 |    0.0671   0.0084   0.9244 |   2.567  4.543  0.910  1.345
   0.0000   0.0000   1.0000 |    0.0039   0.0183   0.9778 |   1.572  5.496  4.578  0.011
   0.0000   1.0000   0.0000 |    0.2288   0.7519   0.0193 |   2.456  2.045  3.917  4.351
   1.0000   0.0000   0.0000 |    0.8898   0.0270   0.0832 |   2.076  3.446  1.858  3.124
   0.0000   1.0000   0.0000 |    0.0269   0.9007   0.0723 |   3.039  2.577  4.580  2.113
   1.0000   0.0000   0.0000 |    0.8051   0.1654   0.0295 |   1.978  2.684  2.797  4.148
   0.0000   1.0000   0.0000 |    0.0143   0.8954   0.0902 |   3.228  1.780  4.552  1.178
   1.0000   0.0000   0.0000 |    0.9595   0.0115   0.0291 |   3.734  4.191  1.410  4.346
   0.0000   1.0000   0.0000 |    0.0569   0.9269   0.0162 |   3.019  1.830  4.968  3.265
   0.0000   1.0000   0.0000 |    0.0200   0.9383   0.0417 |   3.337  2.186  5.758  2.229
   0.0000   0.0000   1.0000 |    0.0061   0.0194   0.9745 |   1.910  4.124  2.908  1.194
   0.0000   0.0000   1.0000 |    0.0304   0.0105   0.9591 |   1.884  4.664  1.581  1.611
   0.0000   0.0000   1.0000 |    0.0283   0.4444   0.5273 |   2.376  2.406  3.279  1.305
   0.0000   0.0000   1.0000 |    0.0364   0.0088   0.9548 |   2.360  5.530  1.742  1.777
```

### Regression Performance
**Regression** is where each input is mapped to a continuous value, rather than a discrete class. This model uses **Linear (Identity)** activation in the output layer and **mean-squared error (MSE)** loss for training.

*Note: Regression tasks generally require more rigorous data preparation and training control. Since output values can have a wider range compared to classification tasks, achieving high performance often demands more careful tuning of the model and training process.*
```python
# Model configuration
nn = NeuralNetwork(
    layers=[4, 12, 12, 3],
    activation=["leaky_relu", "leaky_relu", "id"],
    loss_function="MSE",
    learn_rate=0.001, # Low learning rate to prevent exploding gradients by MSE loss
    lambda_parem=0.003,
    momentum=0.75
)
```
```
                   Expected |                   Predicted | Input Data
 -12.7029  -5.7081  -2.8977 |   -7.7141  -4.5786  -0.7692 |  -0.752 -2.008  0.058  0.689
   7.8434  18.5602   2.2631 |    7.4264  19.0249   0.6353 |   1.991  0.743  2.876 -2.421
  11.4906  10.1924   5.4839 |   13.2474  10.6497   8.1294 |   2.761 -0.804  1.109  3.183
  15.9089 -20.4541 -12.0050 |   15.8087 -21.4759 -12.9103 |  -1.760  1.157 -2.506 -2.679
   2.2022  12.2454  -1.4010 |    4.4052   9.6524  -0.4834 |   1.969 -0.627  1.583 -0.212
 -13.0830   3.6874   4.9318 |  -11.3500  -0.4405   3.4766 |  -2.872  0.705  2.736  1.716
   8.4233   9.0111   6.0307 |    6.8942   5.4828   4.6263 |  -0.982  2.132  1.317 -0.953
   2.6510 -10.0250  -8.7045 |    0.7668 -11.4104  -7.8081 |  -3.006  0.288  0.574 -2.559
  10.3790 -12.0818  -7.1371 |   12.8803 -15.7975  -8.1018 |  -2.422  1.490 -0.463 -2.350
  -8.7059   0.5379  -1.2789 |   -9.6205   1.6387   2.4857 |  -0.490 -1.782 -0.749 -0.772
   4.8650  -7.6227  -3.6861 |    5.3376  -2.3196   1.5519 |   3.126  1.941 -2.518  2.391
  -4.2910  -3.0089  -2.2582 |    2.2409  -0.9641  -1.1598 |   2.955  1.872 -2.089  1.079
  14.3417   4.5132   3.9338 |   12.3643   2.6903   3.6226 |   0.387  2.452  0.071  0.748
  15.4402 -18.2210  -8.7492 |   16.6213 -20.2890 -10.5834 |  -2.577  0.849 -1.967 -2.020
   2.3245  -5.4033  -1.3430 |    3.3757  -3.9180  -1.8757 |  -0.552  0.730 -0.061 -0.509
 -11.2323  -0.4608  -5.4645 |  -10.3634  -0.3082  -5.5105 |  -1.732 -0.288  3.122 -0.949
```
As shown, the neural network performs exceptionally well on the synthetic data. If real-world data exhibits similar relationships between inputs and outputs, the network is likely to perform equally well.

## Synthetic Data Generation
Random data generation is used to simulate a variety of real-world scenarios where the relationship between input features and output targets is not immediately clear. By introducing randomization, we can create a diverse range of datasets that help test the network's ability to generalize and perform well across different situations.

This synthetic data helps in evaluating the model’s performance without the need for a real-world dataset, which can be difficult to acquire or pre-process.

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
Normally, bounded functions like **Sigmoid**, **Tanh**, or **Softmax** activations aren't ideal for regression problems, and unbounded functions like **Leaky ReLU** or **Linear** activations aren't meant for classification.

*But hey, curious about what would happen if you try them anyway? Time to have some fun and experiment! Feel free to break this network with some wild experiments!*

## GPU Acceleration?
This neural network implementation is built using pure **NumPy** utilities, which allows for easy conversion to **CuPy** for GPU acceleration without changing the code structure. By simply replacing **NumPy** with **CuPy**, computations can be offloaded to a CUDA-capable GPU, for faster training for large network and datasets.

However, using GPU acceleration may introduce significant overhead if the neuron count is low or the network has only a few layers, where the benefits of GPU processing may be outweighed by the cost of transferring data to the GPU.

## Resources *& Inspirations :)*
- 3Blue1Brown: [Neural Network Series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=856ZSYGlqoSCdHB2)
- GeeksForGeeks: [What is a Neural Network?](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/)
- TensorFlow: [A Neural Network Playground](https://playground.tensorflow.org/)
- StatQuest with Josh Starmer: [The Softmax Derivative](https://youtu.be/M59JElEPgIg?si=S_ERldGE5K5Jib0E)
- [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
