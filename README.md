
# Simple Neural Network

## Overview
This is the very simple, yet *powerful* example of a fully functional neural network implemented from scratch in Python (with numpy).

## Features
- **Fully connected layers**: Customizable layer sizes.
- **Activation functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax.
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

The **synthetic** data (artificial data created using algorithms) is used to test the model's ability to predict outcomes based on the input features. All tests were conducted with **40,000 training datasets**. The performance is then evaluated on **20,000 unseen test datasets** generated using the same method (*the predictions for 16 unseen test datasets are compared with the expected outputs below*). [How is synthetic data generated?](#synthetic-data-generation)
### Multilabel Classification Performance (Sigmoid activation in last layer + BCE Loss)
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
Accuracy on each output:    96.86%   93.55%   94.00%
                Expected |                Predicted | Input Data
  1.0000  1.0000  1.0000 |   0.9991  0.9945  0.9500 |  -1.983  0.846 -4.952  4.843
  0.0000  1.0000  0.0000 |   0.1397  1.0000  0.0005 |  -2.466 -5.948 -5.055 -2.173
  1.0000  0.0000  0.0000 |   1.0000  0.0689  0.0000 |  -2.377  5.814 -5.809 -0.553
  1.0000  0.0000  1.0000 |   0.9434  0.8368  0.6809 |  -0.574  1.754  1.639 -4.891
  1.0000  0.0000  1.0000 |   0.9909  0.1137  0.9927 |   2.141  3.529  1.113  2.937
  1.0000  0.0000  0.0000 |   0.4521  0.0012  0.0000 |   3.644  4.127 -4.303 -3.109
  0.0000  1.0000  0.0000 |   0.0864  1.0000  0.0003 |  -2.854 -5.954 -3.832 -5.042
  1.0000  0.0000  1.0000 |   1.0000  0.0024  0.9639 |   5.483  3.715  0.737 -1.547
  1.0000  0.0000  0.0000 |   0.8446  0.0213  0.0043 |   1.531  2.075 -1.846 -1.372
  0.0000  1.0000  1.0000 |   0.0073  0.9998  1.0000 |  -3.752  3.874  5.303  1.942
  0.0000  0.0000  1.0000 |   0.0000  0.3164  0.3317 |  -4.201 -0.375  5.935 -0.508
  1.0000  1.0000  1.0000 |   0.9946  0.9975  0.9966 |  -2.170  4.556  1.420 -1.668
  1.0000  1.0000  0.0000 |   0.9999  0.6545  0.0008 |  -2.072  2.306 -2.710 -2.255
  1.0000  0.0000  1.0000 |   0.9952  0.0004  0.7538 |   5.616 -0.949  1.904  0.412
  1.0000  1.0000  1.0000 |   0.9942  0.9988  1.0000 |  -0.822  4.469  5.155  3.986
  1.0000  1.0000  0.0000 |   1.0000  0.9672  0.0139 |  -3.464  2.315 -5.673  0.411
```
### Multiclass Classification Performance (Softmax activation in last layer + CCE Loss)
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
Accuracy on each output:    95.80%   93.84%   97.75%
Overall categorization accuracy:    93.69%
                Expected |                Predicted | Input Data
  0.0000  0.0000  1.0000 |   0.0001  0.0019  0.9980 |   1.707  4.006  2.932  1.359
  0.0000  0.0000  1.0000 |   0.0016  0.1377  0.8607 |   0.169  3.172  3.388  2.011
  0.0000  1.0000  0.0000 |   0.0000  0.9998  0.0002 |   3.153  1.119  5.419  1.899
  0.0000  1.0000  0.0000 |   0.0009  0.2836  0.7155 |   2.872  3.035  3.642  1.178
  1.0000  0.0000  0.0000 |   0.9684  0.0163  0.0153 |   4.683  4.951  3.743  3.266
  0.0000  1.0000  0.0000 |   0.2429  0.7547  0.0024 |   1.870  2.056  3.415  3.424
  0.0000  1.0000  0.0000 |   0.0002  0.6986  0.3011 |   2.311  2.588  4.241  1.306
  1.0000  0.0000  0.0000 |   0.9921  0.0079  0.0000 |   4.800  1.743  2.618  4.171
  0.0000  0.0000  1.0000 |   0.0002  0.0003  0.9995 |   0.313  2.484  0.855  0.778
  0.0000  1.0000  0.0000 |   0.0006  0.9869  0.0126 |   1.773  2.393  4.810  2.371
  0.0000  0.0000  1.0000 |   0.0016  0.0077  0.9907 |   0.889  2.663  2.009  1.349
  0.0000  1.0000  0.0000 |   0.0019  0.9979  0.0003 |   2.830  2.126  5.242  3.158
  0.0000  0.0000  1.0000 |   0.0000  0.0000  1.0000 |   0.035  3.513  0.479  0.179
  1.0000  0.0000  0.0000 |   0.9624  0.0369  0.0007 |   3.362  5.006  3.992  4.595
  1.0000  0.0000  0.0000 |   0.9211  0.0012  0.0777 |   2.898  4.739  0.348  3.234
  0.0000  1.0000  0.0000 |   0.0010  0.9988  0.0002 |   1.363  1.381  5.172  3.616
```
As shown, the neural network performs exceptionally well on the synthetic data. If real-world data exhibits similar relationships between inputs and outputs, the network is likely to perform equally well.

## Synthetic Data Generation
Random data generation is used to simulate a variety of real-world scenarios where the relationship between input features and output targets is not immediately clear. By introducing randomization, we can create a diverse range of datasets that help test the network's ability to generalize and perform well across different situations.

This synthetic data helps in evaluating the model’s performance without the need for a real-world dataset, which can be difficult to acquire or pre-process.

For the above example runs, 4 input features `i1, i2, i3, i4` and 3 output targets `o1, o2, o3` are generated as follows.
```python
# For multilabel classification demonstration
i1, i2, i3, i4 = random.uniform(-6, 6), random.uniform(-6, 6), random.uniform(-6, 6), random.uniform(-6, 6)
o1, o2, o3 = 0.0, 0.0, 0.0

# Define arbitrary relationships between inputs and outputs for demonstration
if i1*i1 - 5*i2 < 2*i1*i3 - i4:           o1 = 1.0
if 4*i1 - 2*i2*i3 + 0.4*i4*i2/i1 < -3*i3: o2 = 1.0
if i1/i4 + 0.3*i2 - 8*i2*i2/i3 < 2*i4:    o3 = 1.0
```
```python
# For multiclass classification demonstration
k = random.randint(0, 2) # Select a random class (0, 1, or 2)
if (k == 0):
    i1, i2, i3, i4 = random.uniform(2, 5), random.uniform(1, 5), random.uniform(0, 4), random.uniform(3, 5)
    o1, o2, o3 = 1.0, 0.0, 0.0
elif (k == 1):
    i1, i2, i3, i4 = random.uniform(1, 4), random.uniform(1, 3), random.uniform(3, 6), random.uniform(0, 5)
    o1, o2, o3 = 0.0, 1.0, 0.0
else:
    i1, i2, i3, i4 = random.uniform(0, 3), random.uniform(2, 6), random.uniform(0, 6), random.uniform(0, 2)
    o1, o2, o3 = 0.0, 0.0, 1.0
```
In all data generation, input features (`i1, i2, i3, i4`) are exposed to some noise to better mimic real-world scenarios.
```python
    input_list.append(self._add_noise([i1, i2, i3, i4], noise=0.2))
    output_list.append([o1, o2, o3])
```
```python
def _add_noise(self, data: list, noise=0.5):
    return [x + random.uniform(-noise, noise) for x in data]
```

## GPU Acceleration?
This neural network implementation is built using pure **NumPy** utilities, which allows for easy conversion to **CuPy** for GPU acceleration without changing the code structure. By simply replacing **NumPy** with **CuPy**, computations can be offloaded to a CUDA-capable GPU, for faster training for large network and datasets.

However, using GPU acceleration may introduce significant overhead if the neuron count is low or the network has only a few layers, where the benefits of GPU processing may be outweighed by the cost of transferring data to the GPU.

## Resources *& Inspirations :)*
- 3Blue1Brown: [Neural Network Series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=856ZSYGlqoSCdHB2)
- GeeksForGeeks: [What is a Neural Network?](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/)
- TensorFlow: [A Neural Network Playground](https://playground.tensorflow.org/)
- StatQuest with Josh Starmer: [The Softmax Derivative](https://youtu.be/M59JElEPgIg?si=S_ERldGE5K5Jib0E)
- [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
