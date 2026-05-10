import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from data_generator import generate_multilabel
from neural_network import NeuralNetwork, Trainer
from neural_network.activations import Tanh, Sigmoid
from neural_network.layers import Dense, Dropout
from neural_network.losses import BCE
from neural_network.optimizers import Adam


def _build():
    nn = NeuralNetwork(
        layers=[
            Dense(10, activation=Tanh()),
            Dense(16, activation=Tanh()),
            Dropout(dropout_rate=0.4),
            Dense(12, activation=Tanh()),
            Dense(3,  activation=Sigmoid())
        ],
        weight_decay=0.001
    )
    nn.build(input_size=4)
    return nn


def test_multilabel():
    nn      = _build()
    X, Y    = generate_multilabel(500, seed=0)
    Y_arr   = np.array(Y)

    preds_before = np.array(nn.forward_batch(X))
    loss_before  = -np.mean(Y_arr * np.log(np.clip(preds_before, 1e-12, 1 - 1e-12))
                            + (1 - Y_arr) * np.log(np.clip(1 - preds_before, 1e-12, 1 - 1e-12)))

    trainer = Trainer(nn, loss_function=BCE(), optimizer=Adam(learn_rate=0.02), verbose=False)
    trainer.train(X, Y, epoch=30, batch_size=64, show_loss=False)

    preds_after = np.array(nn.forward_batch(X))
    loss_after  = -np.mean(Y_arr * np.log(np.clip(preds_after, 1e-12, 1 - 1e-12))
                           + (1 - Y_arr) * np.log(np.clip(1 - preds_after, 1e-12, 1 - 1e-12)))

    assert loss_after < loss_before, \
        f"multilabel loss did not improve: {loss_before:.4f} -> {loss_after:.4f}"


def demo_multilabel():
    nn      = _build()
    trainer = Trainer(nn, loss_function=BCE(), optimizer=Adam(learn_rate=0.02))

    X_train, Y_train = generate_multilabel(2000)
    X_test,  Y_test  = generate_multilabel(1000)
    X_show,  Y_show  = generate_multilabel(16)

    trainer.train(X_train, Y_train, epoch=30, batch_size=64)
    nn.metrics.check_accuracy(X_test, Y_test)
    nn.metrics.compare_predictions(X_show, Y_show)


if __name__ == "__main__":
    demo_multilabel()
    input("Press any key to exit.")
