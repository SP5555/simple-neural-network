
# Simple Neural Network

## Overview
This is the very simple, yet *powerful* example of a fully functional neural network implemented from scratch in Python (with numpy).

## Features
- **Fully connected layers**: Customizable layer sizes.
- **Activation functions**: Sigmoid, Tanh, ReLU, Leaky ReLU.
- **Loss functions**: Mean-Squared Error (MSE) for regression, Binary Cross-Entropy (BCE) for multilabel classification, Categorical Cross-Entropy for (CCE or MCE) for multiclass classification.
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

The **synthetic** data (artificial data created using algorithms) is used to test the model's ability to predict outcomes based on the input features. All tests were conducted with **40,000 training datasets**. The performance is then evaluated on **20,000 unseen test datasets** generated using the same method (*the predictions for 16 unseen test datasets are compared with the expected outputs below*).
### Multilabel Classification Performance (Sigmoid activation in last layer + BCE Loss)
Model configuration:
```python
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
Accuracy on 20000 samples:    96.81%   93.44%   93.25%
                Expected |                Predicted | Input Data
  0.0000  1.0000  0.0000 |   0.0001  1.0000  0.0001 |   0.037 -5.637 -5.111  0.110
  0.0000  0.0000  1.0000 |   0.0000  0.0029  0.9967 |  -0.477 -5.387  1.922  2.747
  0.0000  1.0000  0.0000 |   0.0004  0.9989  0.7204 |  -4.734 -0.439 -0.028  1.521
  0.0000  0.0000  1.0000 |   0.0001  0.5250  0.2523 |  -2.700  0.245  4.635  1.071
  0.0000  0.0000  1.0000 |   0.0007  0.0000  0.9445 |   4.674 -2.236  2.094  1.628
  0.0000  0.0000  0.0000 |   0.0000  0.0960  0.1013 |   4.593 -0.452 -3.764 -4.956
  0.0000  0.0000  1.0000 |   0.4673  0.0000  0.9518 |   2.477 -3.131  3.174 -3.134
  1.0000  1.0000  1.0000 |   0.9993  0.9739  0.9976 |  -1.701  4.501  1.510  5.049
  1.0000  0.0000  1.0000 |   0.9551  0.1955  0.8019 |   0.308  1.016  4.879  1.107
  0.0000  0.0000  1.0000 |   0.1546  0.0002  0.8470 |   0.898 -2.876  3.908 -4.511
  1.0000  1.0000  1.0000 |   0.9993  0.9997  0.6687 |  -3.616 -1.222 -5.651  0.106
  0.0000  1.0000  0.0000 |   0.0007  0.6477  0.6900 |   1.400 -1.893 -0.865  5.940
  0.0000  0.0000  1.0000 |   0.0075  0.0007  0.5747 |   3.730 -1.057  1.342 -1.115
  0.0000  1.0000  0.0000 |   0.0049  1.0000  0.0007 |  -2.908 -4.495 -2.989 -2.056
  1.0000  0.0000  0.0000 |   0.9943  0.0558  0.0008 |   0.231  2.978 -5.318 -1.276
  0.0000  0.0000  1.0000 |   0.0000  0.0004  0.9900 |  -3.041 -3.689  4.779  1.332
```
### Multiclass Classification Performance (Softmax activation in last layer + CCE Loss)
Model configuration:
```python
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
Accuracy on 20000 samples:    96.32%   92.32%   95.34%
                Expected |                Predicted | Input Data
  1.0000  0.0000  0.0000 |   0.3516  0.6478  0.0006 |   3.363  2.095  3.635  3.580
  0.0000  1.0000  0.0000 |   0.0022  0.7856  0.2122 |   3.412  2.715  4.084  0.062
  1.0000  0.0000  0.0000 |   0.9946  0.0052  0.0002 |   3.739  3.324  2.445  4.157
  0.0000  1.0000  0.0000 |   0.3877  0.6114  0.0009 |   2.363  2.480  3.673  4.711
  0.0000  0.0000  1.0000 |   0.2633  0.1037  0.6331 |   1.981  2.774  1.768  1.979
  0.0000  0.0000  1.0000 |   0.0005  0.0236  0.9759 |   0.699  3.196  3.101  1.220
  0.0000  1.0000  0.0000 |   0.5116  0.4883  0.0000 |   3.938  1.687  4.129  4.734
  1.0000  0.0000  0.0000 |   0.9956  0.0043  0.0001 |   2.819  2.069  1.427  4.554
  1.0000  0.0000  0.0000 |   0.9964  0.0012  0.0025 |   4.797  4.244  1.935  3.031
  0.0000  0.0000  1.0000 |   0.0002  0.0784  0.9215 |   2.709  4.568  4.968  0.377
  0.0000  0.0000  1.0000 |   0.0001  0.0267  0.9732 |   0.889  3.925  4.859  0.284
  1.0000  0.0000  0.0000 |   0.9311  0.0689  0.0001 |   4.335  2.300  3.289  4.154
  0.0000  1.0000  0.0000 |   0.0019  0.2777  0.7204 |   1.389  2.367  3.903  0.118
  0.0000  0.0000  1.0000 |   0.0007  0.0129  0.9864 |   0.645  2.867  1.902  0.523
  0.0000  0.0000  1.0000 |   0.0199  0.5622  0.4179 |   1.199  2.040  2.578  1.619
  1.0000  0.0000  0.0000 |   0.9376  0.0621  0.0003 |   3.538  1.747  2.266  3.482
```
As shown, the neural network performs exceptionally well on the synthetic data. If real-world data exhibits similar relationships between inputs and outputs, the network is likely to perform equally well.

## GPU Acceleration?
This neural network implementation is built using pure **NumPy** utilities, which allows for easy conversion to **CuPy** for GPU acceleration without changing the code structure. By simply replacing **NumPy** with **CuPy**, computations can be offloaded to a CUDA-capable GPU, for faster training for large network and datasets.

However, using GPU acceleration may introduce significant overhead if the neuron count is low or the network has only a few layers, where the benefits of GPU processing may be outweighed by the cost of transferring data to the GPU.

## Resources *& Inspirations :)*
- 3Blue1Brown: [Neural Network Series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=856ZSYGlqoSCdHB2)
- GeeksForGeeks: [What is a Neural Network?](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/)
- TensorFlow: [A Neural Network Playground](https://playground.tensorflow.org/)
- StatQuest with Josh Starmer: [The Softmax Derivative](https://youtu.be/M59JElEPgIg?si=S_ERldGE5K5Jib0E)
- [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
