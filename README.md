
# Simple Neural Network

## Overview
This is the very simple, yet powerful example of a fully functional neural network implemented from scratch in Python (with numpy).

## Features
- **Fully connected layers**: Customizable layer sizes.
- **Activation functions**: Sigmoid, Tanh, ReLU, Leaky ReLU.
- **Loss functions**: Mean-Squared Error (MSE) for regression, Binary Cross-Entropy (BCE) for binary classification. (MCE not available yet)
- **Training Algorithm**: Gradient descent.

## Installation Instructions
Run the following commands to install it locally. Require Python `3.0` and `pip`.
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
	loss_function="BCE", # for binary classification
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
nn.check_accuracy_binary_classification(
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
### Dataset Generation
The dataset consists of 4 floating-point input features and 2 floating-point output labels. Here's how a dataset instance is generated:
```python
i1: float = random.uniform(-6, 6)
i2: float = random.uniform(-6, 6)
i3: float = random.uniform(-6, 6)
i4: float = random.uniform(-6, 6)
o1: float = 0.0
o2: float = 0.0

# Make your own Data Here
if i1*i1 - 5*i2 < 2*i1*i3 - i4: o1 = 1.0
if 4*i1 - 2*i2*i3 + 0.4*i4/i1 < -3*i3: o2 = 1.0

# noise for inputs
i1  +=  random.uniform(-0.2, 0.2)
i2  +=  random.uniform(-0.2, 0.2)
i3  +=  random.uniform(-0.2, 0.2)
i4  +=  random.uniform(-0.2, 0.2)
```
The synthetic dataset is used for testing the model's ability to predict binary outcomes based on the input features.
### Training Results
After training on **40,000 datasets** using the **Binary Cross-Entropy (BCE) loss function**, the following accuracy was achieved on **20,000 unseen test datasets** generated using the same method (*the predictions for 16 unseen test datasets are compared against the expected outputs*):
```
Accuracy on 20000 samples:    96.73%   95.62%
        Expected |        Predicted | Input Data
  1.0000  1.0000 |   1.0000  0.9941 |   1.235  5.677  5.328 -4.983
  1.0000  1.0000 |   0.7976  1.0000 |  -4.581 -2.316 -4.042  1.983
  0.0000  1.0000 |   0.0000  0.9998 |  -5.748  0.900  4.935  5.049
  1.0000  0.0000 |   0.9941  0.0460 |   2.240 -0.214  3.774 -2.981
  0.0000  1.0000 |   0.0033  0.5099 |   1.373  0.082 -3.187  1.617
  0.0000  0.0000 |   0.0000  0.0018 |   3.366  0.950 -3.816 -4.563
  0.0000  1.0000 |   0.0007  0.9967 |   2.236 -1.630 -4.260 -5.675
  0.0000  0.0000 |   0.0000  0.0430 |   5.346 -1.955 -2.449  2.532
  0.0000  0.0000 |   0.0012  0.0030 |  -1.655 -2.013  3.774 -5.385
  1.0000  1.0000 |   0.9944  0.9998 |  -4.875 -0.696 -4.410  2.508
  1.0000  0.0000 |   1.0000  0.5578 |  -3.883  4.866 -2.117 -5.376
  0.0000  1.0000 |   0.0003  0.9999 |   2.618 -3.652 -4.444 -3.326
  1.0000  0.0000 |   1.0000  0.7164 |  -5.537  5.203 -3.914 -4.519
  1.0000  0.0000 |   0.9216  0.0027 |   3.670 -3.443  4.228 -5.562
  0.0000  0.0000 |   0.0065  0.0003 |   3.298 -4.929  3.002 -4.985
  0.0000  1.0000 |   0.0004  0.7648 |   1.810 -0.481 -3.386 -0.003
```
As shown, the neural network performs exceptionally well on the synthetic data. If real-world data exhibits similar relationships between inputs and outputs, the network is likely to perform equally well.

## GPU Acceleration?
This neural network implementation is built using pure **NumPy** utilities, which allows for easy conversion to **CuPy** for GPU acceleration without changing the code structure. By simply replacing **NumPy** with **CuPy**, computations can be offloaded to a CUDA-capable GPU, for faster training for large network and datasets.

However, using GPU acceleration may introduce significant overhead if the neuron count is low or the network has only a few layers, where the benefits of GPU processing may be outweighed by the cost of transferring data to the GPU.

## Resources *& Inspirations :)*
- 3Blue1Brown: [Neural Network Series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=856ZSYGlqoSCdHB2)
- GeeksForGeeks: [What is a Neural Network?](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/)

## Future Improvements
**Multiclass Cross-Entropy (MCE) Loss Function**: The implementation of a multiclass cross-entropy loss function is planned to extend the network's capabilities for multi-class classification tasks.