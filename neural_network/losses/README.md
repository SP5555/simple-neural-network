# Loss Functions
Loss functions measure how well a model's predictions align with the target values. They get applied to each neuron in the output layer, and the whole point of training a neural network is to adjust the parameters so that the loss gets as small as possible. The better we can get each neuron to predict with less loss, the more accurate the model becomes. The entire realm of Linear Algebra and Calculus has been summoned to minimize this loss value.

## Mean Squared Error - L2 Loss (MSE)
A commonly used loss function. As the name suggests, it is just the square of the error. The squaring effect diminishes small errors (values less than 1) but causes large errors to explode, making this both a strength and a weakness of the **Mean Squared Error** loss function. Due to its ability to deal with errors of all sizes (assuming we have infinite hardware power to handle extremely large squared values), it is most suited for regression tasks.

$$Loss_{MSE}=( y_{actual} -y_{predicted})^{2}$$

## Mean Absolute Error - L1 Loss (MAE)
**Mean Absolute Error** provides a linear loss, making it less sensitive to large errors compared to the quadratic loss of **MSE**, which can amplify large deviations. This prevents the issue of exploding errors seen in the **MSE**. At the same time, the linearity can't differentiate between overestimation and underestimation, which is a well known limitation. By treating all errors equally, **MAE** causes the large errors have the same impact as small errors, potentially masking important issues in training. 

$$Loss_{MAE}=|y_{actual} -y_{predicted}|$$

## Huber Loss
**MSE** too aggressive? **MAE** too forgiving? **Huber Loss** got your back. Controlled by the hyperparameter $\delta$, **Huber Loss** behaves like **MSE** for errors within $\delta$ and **MAE** for larger errors. This fixes the **MSE**'s oversensitiveness towards extreme values (outliers) and **MAE**'s linearity, offering a good balance. However, choosing the right $\delta$ can be tricky depending on the data.

$$Loss_{Huber} =\begin{cases}
\frac{1}{2} (y_{actual} -y_{predicted})^{2} & \text{if}\ |y_{actual} -y_{predicted}| \leq \delta\\
\delta(|y_{actual} -y_{predicted}|-\frac{1}{2}\delta)& \text{if}\ |y_{actual} -y_{predicted}| > \delta\\
\end{cases}$$

## Binary Cross-Entropy (BCE)
When **MSE** struggles with small error values (less than 1) due to squaring, which diminishes those values, **Binary Cross-Entropy** is the answer. The logarithmic functions help amplify the error between 0 and 1, making it much more effective at penalizing incorrect predictions in this tiny range. This gives **BCE** an edge, especially in binary classification tasks where the model's predictions need to be confident (either 0 or 1).

$$Loss_{BCE}=-y_{actual}\log(y_{predicted})-(1-y_{actual})\log(1-y_{predicted})$$

## Categorial Cross-Entropy (CCE)
While **BCE** penalizes both 1s for not being 0 and 0s for not being 1, **Categorical Cross-Entropy** (CCE) focuses only on punishing the 0s when they should be 1. Designed to work with one-hot encoded values *together* with the **Softmax** activation function, **CCE** can guide a neural network to categorize data accurately by heavily penalizing incorrect 0 predictions. Once low raw scores become high enough, the **Softmax** activation will convert them into a high probability distribution.

$$Loss_{CCE}=-y_{actual}\log(y_{predicted})$$
