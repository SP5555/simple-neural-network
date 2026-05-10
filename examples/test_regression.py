import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from data_generator import generate_regression
from neural_network import NeuralNetwork, Trainer
from neural_network.activations import PReLU, Tanh, Swish, Linear
from neural_network.layers import Dense, BatchNorm, Dropout
from neural_network.losses import Huber
from neural_network.optimizers import Adam


def _build():
    nn = NeuralNetwork(
        layers=[
            Dense(12, activation=PReLU()),
            Dense(16, activation=Tanh()),
            BatchNorm(),
            Dropout(dropout_rate=0.4),
            Dense(12, activation=Swish()),
            Dense(3,  activation=Linear())
        ],
        weight_decay=0.001
    )
    nn.build(input_size=4)
    return nn


def test_regression():
    nn      = _build()
    X, Y    = generate_regression(500, seed=0)
    Y_arr   = np.array(Y)

    preds_before = np.array(nn.forward_batch(X))
    loss_before  = np.mean(np.square(preds_before - Y_arr))

    trainer = Trainer(nn, loss_function=Huber(delta=2.5), optimizer=Adam(learn_rate=0.01), verbose=False)
    trainer.train(X, Y, epoch=30, batch_size=64, show_loss=False)

    preds_after = np.array(nn.forward_batch(X))
    loss_after  = np.mean(np.square(preds_after - Y_arr))

    assert loss_after < loss_before, \
        f"regression loss did not improve: {loss_before:.4f} -> {loss_after:.4f}"


def demo_regression():
    nn      = _build()
    trainer = Trainer(nn, loss_function=Huber(delta=2.5), optimizer=Adam(learn_rate=0.01))

    X_train, Y_train = generate_regression(2000)
    X_test,  Y_test  = generate_regression(1000)
    X_show,  Y_show  = generate_regression(16)

    trainer.train(X_train, Y_train, epoch=30, batch_size=64)
    nn.metrics.check_accuracy(X_test, Y_test)
    nn.metrics.compare_predictions(X_show, Y_show)


if __name__ == "__main__":
    demo_regression()
    input("Press any key to exit.")
