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

Noticed something? Yes. **RMSprop** is like **SGD**. It is also a "drunk-man" optimizer but with a little more sense, knowing how far it should step.

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

## Adaptive Moment Estimation (Adam)

**Momentum** smooths out the updates with *moving average of gradients*. However, because the learning rate remains constant, it can overshoot, especially in abrupt "steep" or "narrow" regions of the loss landscape. $m$ is known as the *first moment*:

```math
m_{t} =\beta _{1} m_{t-1} +( 1-\beta _{1})\frac{\partial L}{\partial w_{t}}
```

**RMSprop**, on the other hand, does a good job at adapting the learn rate dynamically with the *moving average of squared gradients*. But it lacks directional awareness (due to squaring of the loss gradient), making it behave similar to "drunk-man" wandering in random directions but with controlled step sizes. $v$ is known as the *second moment*:

```math
v_{t} =\beta _{2} v_{t-1} +( 1-\beta _{2})\left(\frac{\partial L}{\partial w_{t}}\right)^{2}
```

What does **Adam** do? It combines the strength of **Momentum** (smooth updates) and **RMSprop** (adaptive learn rate). One thing to note. **Adam** applies "bias correction" on $m_{t}$ and $v_{t}$ first.

```math
\widehat{m_{t}} =\frac{m_{t}}{1-\beta _{1}^{t}}
,\quad
\widehat{v_{t}} =\frac{v_{t}}{1-\beta _{2}^{t}}
```

Unfortunately, This "bias correction" is naturally not so intuitive. But here is a dumb way to understand it intuitively. Since $m_t$ and $v_t$ both initialize as zero, the moving average nature requires some iterations for them to become strong enough to actually start working. the factors $\frac{1}{1-\beta _{1}^{t}}$ and $\frac{1}{1-\beta _{2}^{t}}$ "warm up" the early training cycles so that $\widehat{m_{t}}$ and $\widehat{v_{t}}$ are relatively strong to produce noticeable updates.
- $\widehat{m_{t}}$ influences the direction and magnitude of the update step as in **Momentum**.
- $\widehat{v_{t}}$ controls the learn rate adaptively as in **RMSprop**.

Then weights are updated as follows:

```math
w_{t+1} =w_{t} -\frac{\eta }{\sqrt{\widehat{v_{t}} +\epsilon }}\widehat{m_{t}}
```

**Adam** takes the best of both **Momentum** and **RMSprop** worlds. It's so powerful and efficient that you can pretty much slap it into any random network, whatever it is, and it'll still give you good results.