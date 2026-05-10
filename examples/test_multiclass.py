import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from data_generator import generate_multiclass
from neural_network import NeuralNetwork, Trainer
from neural_network.activations import PReLU, Tanh, Swish, Softmax
from neural_network.layers import Dense, Dropout
from neural_network.losses import CCE
from neural_network.optimizers import Adam


def _build():
    nn = NeuralNetwork(
        layers=[
            Dense(12, activation=PReLU()),
            Dense(16, activation=Tanh()),
            Dropout(dropout_rate=0.4, batch_wise=True),
            Dense(12, activation=Swish()),
            Dropout(dropout_rate=0.4),
            Dense(3,  activation=Softmax())
        ],
        weight_decay=0.001
    )
    nn.build(input_size=6)
    return nn


def test_multiclass():
    nn      = _build()
    X, Y    = generate_multiclass(500, seed=0)
    Y_arr   = np.array(Y)

    preds_before = np.array(nn.forward_batch(X))
    loss_before  = -np.mean(Y_arr * np.log(np.clip(preds_before, 1e-12, 1.0)))

    trainer = Trainer(nn, loss_function=CCE(), optimizer=Adam(learn_rate=0.02), verbose=False)
    trainer.train(X, Y, epoch=30, batch_size=64, show_loss=False)

    preds_after = np.array(nn.forward_batch(X))
    loss_after  = -np.mean(Y_arr * np.log(np.clip(preds_after, 1e-12, 1.0)))

    assert loss_after < loss_before, \
        f"multiclass loss did not improve: {loss_before:.4f} -> {loss_after:.4f}"


def demo_multiclass():
    nn      = _build()
    trainer = Trainer(nn, loss_function=CCE(), optimizer=Adam(learn_rate=0.02))

    X_train, Y_train = generate_multiclass(2000)
    X_test,  Y_test  = generate_multiclass(1000)
    X_show,  Y_show  = generate_multiclass(16)

    trainer.train(X_train, Y_train, epoch=30, batch_size=64)
    nn.metrics.check_accuracy(X_test, Y_test)
    nn.metrics.compare_predictions(X_show, Y_show)


if __name__ == "__main__":
    demo_multiclass()
    input("Press any key to exit.")
