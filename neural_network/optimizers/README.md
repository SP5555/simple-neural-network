# Optimizers
After activation functions and loss functions compute gradients, what's next? Gradients points in the direction of the steepest climb. In this case, toward higher loss. So, intuitively speaking, to minimize the loss, parameters must move in the opposite direction of the gradient. But how so? Different optimizers have different rules on how weights move in that opposite direction.

- $\eta$ is the **learn rate**, a dominant hyperparameter that exists in all optimization algorithms. It controls how fast or slow the parameter updates. Typically, $\eta$ values less than 0.01 are preferred. It can go as low as 0.0001 in some cases. That said, take it with a grain of salt. The optimal $\eta$ value really depends on what you're doing.

- $\epsilon$ is a very small positive number. You will find it occasionally down here. Its only purpose is to prevent some problematic math operations like dividing by zero.

## Stochastic Gradient Descent (SGD)

The vanilla "drunk-man" optimizer. There is not much fancy stuff going on here. The negative of the gradient is scaled with the **learn rate** and applied directly to the weights. It is called "drunk-man" because it might wander around while slowly making its way towards the local minimum. This is due to the batch-nature of the implemented training algorithm.

$$w_{t+1} =w_{t} -\eta \frac{\partial L}{\partial w_{t}}$$

## Momentum

In physics, objects with momentum continue to maintain that momentum unless external forces act upon them. Similarly, in neural networks, when parameters are moving toward a local minimum to minimize the loss, the momentum technique gives them the ability to "glide." This "gliding" helps them escape high-loss plateaus faster, allowing them to reach the local minimum more efficiently. It introduces the concept of "velocity" for each parameter in the model. The velocity is updated as follows:

$$v_{t} =\beta v_{t-1} +( 1-\beta ) \frac{\partial L}{\partial w_{t}}$$

where $\beta$ is the momentum strength, ranging between 0 and 1. A value of 0 disables the "gliding" behavior while 1 results in full "gliding" with no update. Ideally, $\beta$ is set between 0.5 and 0.8.

And the calculated velocity is applied to the weights as follows:

$$w_{t+1} =w_{t} -\eta v_{t}$$

## Root Mean Square Propagation (RMSprop)

The idea behind **RMSprop** is similar to **Momentum**. Instead of using velocity term to update the weights directly, **RMSprop** maintains an exponentially decaying average of past squared gradients:

```math
v_{t} =\beta v_{t-1} +( 1-\beta )\left(\frac{\partial L}{\partial w_{t}}\right)^{2}
```

and re-scales the **learn rate** with it.

```math
\eta _{scaled} =\frac{\eta }{\sqrt{v_{t} +\epsilon }}
```

This slow down updates for frequently changing weights while speeding up updates for more stable weights. This helps distribute learning across the network rather than letting a few strong weights dominate the training process.

```math
w_{t+1} =w_{t} -\eta _{scaled}\frac{\partial L}{\partial w_{t}}
```
## Adaptive Gradient (AdaGrad)

**Adaptive Gradient** has a special term that accumulates the squared of gradients (meaning it is never negative) that grows slowly throughout the training.

```math
G=\sum _{i=0}^{t}\left(\frac{\partial L}{\partial w_{t}}\right)^{2}
```

And similar to **RMSprop**, it re-scales the **learn rate** with it.

```math
\eta _{scaled} =\frac{\eta }{\sqrt{G+\epsilon }}
```

Weights are then updated with dynamically adjusted learn rate.

```math
w_{t+1} =w_{t} -\eta _{scaled}\frac{\partial L}{\partial w_{t}}
```

You might have already guessed it at this point. Since $G$ is non-decreasing and continuously grows, the scaled learn rate $\eta _{scaled}$ will eventually perish to dust (I mean, it will shrink to near zero). This means, at some point, the **AdaGrad** will cause the entire network to stop learning.

Is it a good thing or a bad thing? The answer is, both. If training halts too early, the network may underfit and perform poorly. However, since learn rate gradually dies out, it works well when the network is already close to convergence, where only small parameter adjustments are needed.
