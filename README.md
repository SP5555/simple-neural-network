
# Simple Neural Network

## Overview
This is the a very simple, yet powerful example of a fully functional neural network implemented from scratch in Python (with numpy).

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
if i1*i1 - 5*i2 < 2*i3 - i4: o1 = 1.0
if 4*i1 - 2*i2 + i4/2 < - 3*i3: o2 = 1.0

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
Accuracy on 20000 samples:    96.23%   98.17%
        Expected |        Predicted | Input Data
  1.0000  1.0000 |   0.9272  1.0000 |  -2.055  4.687 -4.179  4.528
  0.0000  1.0000 |   0.0868  1.0000 |  -5.947  5.216  0.501  1.213
  0.0000  0.0000 |   0.0000  0.0001 |  -1.468 -2.356  4.060  5.329
  1.0000  0.0000 |   0.6193  0.0008 |   2.767  1.180  2.718  3.614
  1.0000  1.0000 |   0.9698  1.0000 |   2.942  5.243 -4.643 -1.748
  0.0000  1.0000 |   0.0000  1.0000 |  -5.256 -0.349 -4.154 -1.312
  1.0000  0.0000 |   0.9729  0.0761 |   1.112  1.891  1.989 -2.678
  0.0000  1.0000 |   0.0000  1.0000 |  -4.822 -2.160 -2.762  3.010
  0.0000  1.0000 |   0.0118  0.3926 |   2.351 -0.502 -3.174 -4.786
  0.0000  1.0000 |   0.4135  0.9989 |  -4.249  0.338  4.936 -4.054
  0.0000  0.0000 |   0.0000  0.0000 |   2.293 -5.927 -0.383 -4.063
  0.0000  1.0000 |   0.0062  1.0000 |  -1.086  1.176 -2.135  4.289
  1.0000  1.0000 |   0.9979  1.0000 |  -3.005  5.309 -3.153 -5.041
  0.0000  1.0000 |   0.2460  1.0000 |   1.233  2.271 -6.049 -1.169
  0.0000  0.0000 |   0.0126  0.0001 |   6.099  3.830  1.288  2.636
  0.0000  1.0000 |   0.0119  1.0000 |  -4.233  2.629 -0.189  3.391
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