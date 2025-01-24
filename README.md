
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
- [Resources *& Inspirations*](#resources--inspirations-)

## Features
- **Fully connected layers**: Customizable layer sizes.
- **Activation functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax.
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
* `activation_hidden`: Activation function for hidden layers (E.g., `"relu"`, `"sigmoid"`).
* `activation_output`: Activation function for the output layer.
* `loss_function`: Loss function for training (E.g., `"MSE"`, `"BCE"`)
* `learn_rate`: Learning rate for gradient descent.
* `lambda_parem`: Regularization strength to prevent overfitting.
* `momentum`: Momentum to accelerate convergence.

Example with added parameters:
```python
nn = NeuralNetwork(
	layers=[4, 8, 6, 6, 2],
	activation_hidden='leaky_relu',
	activation_output='sigmoid', # for classification
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
    activation_hidden='leaky_relu',
    activation_output='sigmoid',
    loss_function="BCE",
    learn_rate=0.08,
    lambda_parem=0.003,
    momentum=0.75
)
```
```
Accuracy on 20,000 samples
Accuracy on each output:    96.53%   93.27%   93.16%
                   Expected |                   Predicted | Input Data
   1.0000   0.0000   1.0000 |    0.9984   0.0001   0.9179 |   3.366 -2.848  2.007 -4.548
   0.0000   1.0000   0.0000 |    0.0000   0.9996   0.0000 |   2.148 -5.773 -4.391 -6.048
   1.0000   0.0000   1.0000 |    1.0000   0.0617   0.6922 |   4.885  2.000  2.060 -1.791
   0.0000   0.0000   1.0000 |    0.0000   0.0000   1.0000 |  -1.419 -5.664  6.003  2.735
   0.0000   1.0000   0.0000 |    0.0000   0.5914   0.0014 |   4.802 -3.247 -2.441 -0.738
   0.0000   0.0000   1.0000 |    0.0016   0.0578   0.9980 |  -1.842 -2.909  1.368  3.506
   1.0000   1.0000   1.0000 |    1.0000   0.9975   0.8849 |  -4.672  4.406  0.315  3.947
   1.0000   1.0000   0.0000 |    0.8399   0.9991   0.0001 |  -2.352 -2.247 -5.126 -5.468
   0.0000   0.0000   1.0000 |    0.0000   0.0002   1.0000 |   0.543 -4.937  3.004  4.667
   0.0000   1.0000   0.0000 |    0.4893   0.9983   0.0001 |  -2.157 -2.923 -4.347 -4.040
   1.0000   0.0000   0.0000 |    0.9448   0.0151   0.0001 |   0.837  4.246 -3.149  5.326
   0.0000   0.0000   1.0000 |    0.0000   0.0075   0.9998 |  -1.498 -4.266  1.923  4.073
   0.0000   1.0000   0.0000 |    0.0009   0.9980   0.0001 |  -0.231 -4.627 -3.432  2.662
   1.0000   0.0000   0.0000 |    0.9913   0.0468   0.0000 |   0.715  2.829 -3.229 -2.805
   1.0000   0.0000   1.0000 |    1.0000   0.8350   0.9903 |   4.438  4.054  2.747 -3.378
   1.0000   1.0000   1.0000 |    1.0000   0.9480   0.9939 |   3.899  5.023  5.022 -5.820
```

### Multiclass Classification Performance
**Multiclass classification** is where each input belongs to exactly one class. This model uses **Softmax** activation in the output layer and **categorial cross-entropy (CCE)** loss for training.
```python
# Model configuration
nn = NeuralNetwork(
    layers=[4, 12, 12, 3],
    activation_hidden='leaky_relu',
    activation_output='softmax',
    loss_function="CCE",
    learn_rate=0.08,
    lambda_parem=0.003,
    momentum=0.75
)
```
```
Accuracy on 20,000 samples
Accuracy on each output:    95.64%   93.92%   97.87%
Overall categorization accuracy:    93.72%
                   Expected |                   Predicted | Input Data
   0.0000   0.0000   1.0000 |    0.0001   0.0001   0.9998 |   0.341  3.343  0.553  1.010
   0.0000   0.0000   1.0000 |    0.0000   0.0001   0.9999 |   2.182  4.255  1.848  1.188
   0.0000   1.0000   0.0000 |    0.0055   0.9937   0.0007 |   3.344  2.721  4.642  3.031
   0.0000   0.0000   1.0000 |    0.0003   0.2822   0.7175 |   1.445  2.451  3.130  1.556
   0.0000   0.0000   1.0000 |    0.0000   0.0000   1.0000 |   0.135  4.795  0.339  0.685
   0.0000   1.0000   0.0000 |    0.5241   0.4752   0.0008 |   2.695  2.700  3.229  3.915
   0.0000   1.0000   0.0000 |    0.0036   0.9937   0.0027 |   1.413  2.182  4.281  3.208
   0.0000   1.0000   0.0000 |    0.0011   0.9986   0.0004 |   3.879  1.266  4.472  2.196
   0.0000   0.0000   1.0000 |    0.0000   0.0043   0.9957 |   1.337  4.233  4.815  0.856
   0.0000   1.0000   0.0000 |    0.3101   0.6896   0.0003 |   2.992  2.367  3.448  3.808
   1.0000   0.0000   0.0000 |    0.9980   0.0015   0.0005 |   2.356  2.394  0.381  4.857
   0.0000   0.0000   1.0000 |    0.0000   0.0020   0.9980 |   1.852  4.118  4.989  0.246
   0.0000   0.0000   1.0000 |    0.0001   0.0000   0.9999 |   2.511  4.188  0.962  1.071
   0.0000   1.0000   0.0000 |    0.0000   0.9489   0.0511 |   0.955  3.031  5.394  2.472
   1.0000   0.0000   0.0000 |    0.9960   0.0039   0.0001 |   3.260  1.211  0.350  3.475
   1.0000   0.0000   0.0000 |    0.9844   0.0156   0.0000 |   4.627  1.746  1.827  3.627
```

### Regression Performance
**Regression** is where each input is mapped to a continuous value, rather than a discrete class. This model uses **Linear (Identity)** activation in the output layer and **mean-squared error (MSE)** loss for training.

*Note: Regression tasks generally require more rigorous data preparation and training control. Since output values can have a wider range compared to classification tasks, achieving high performance often demands more careful tuning of the model and training process.*
```python
# Model configuration
nn = NeuralNetwork(
    layers=[4, 12, 12, 3],
    activation_hidden="tanh",
    activation_output="id",
    loss_function="MSE",
    learn_rate=0.08,
    lambda_parem=0.003,
    momentum=0.75
)
```
```
                   Expected |                   Predicted | Input Data
 -16.4260  10.8446   3.5806 |  -12.7717   7.0471   0.4925 |   2.456 -0.950 -0.884 -1.324
 -12.1671  -5.3751  -6.7291 |  -16.3188  -7.3254 -11.6916 |  -1.511 -1.477  1.965  0.364
  13.4816 -18.2943 -11.1137 |    7.8558 -10.8403  -6.6613 |  -1.337  0.436 -1.930 -3.258
 -23.3492 -14.2575 -10.5762 |  -16.6960  -7.5671 -11.8593 |  -1.973 -2.586  1.930  0.756
  13.8075   7.2677   6.6929 |   13.8506   6.4844   4.2223 |   1.773  1.655 -0.099  2.316
   5.2817  -3.2575   3.4125 |    6.5779  -6.5257   2.4601 |   0.210  0.824 -1.281  2.394
   1.5152   2.7160  -2.2921 |   -0.0808   4.8274  -1.9845 |   1.628 -1.980  1.732  2.267
 -19.5268 -10.2634 -14.3327 |  -16.5914  -7.2105 -12.5115 |  -1.445 -2.659  2.036 -0.415
  13.1467  17.1063   3.0304 |    6.1894   7.5210   2.6596 |   2.062 -0.403  2.645  0.543
  19.6225 -24.7282  -8.6363 |   15.4704 -21.8385  -7.6052 |  -2.055  0.671 -3.108 -0.103
   4.1203   1.2206   6.8605 |    0.9498   1.7880   2.5967 |   0.657 -0.332 -0.609  2.940
  -0.3894   4.6636 -13.6448 |   -5.4567   7.2505  -7.0076 |   2.617 -2.183  2.382 -0.287
   1.8029   7.6694  10.9483 |    2.2082   7.0463  10.0545 |  -2.047  2.566  1.544  2.003
  21.6373  16.9655  13.0360 |   19.1426  19.9064  12.5229 |   0.842  2.660  0.792  2.803
   1.4599  -6.5495  -4.8901 |   -2.8113  -3.2333  -5.7527 |  -0.143  0.377 -1.738 -1.841
  22.0828  29.9685  14.6448 |   19.1210  19.9341  12.4522 |   1.821  2.094  2.873 -0.311
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
