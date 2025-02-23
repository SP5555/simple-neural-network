from data_generator import DataGenerator
from neural_network import NeuralNetwork, Trainer
from neural_network.activations import *
from neural_network.layers import *
from neural_network.losses import *
from neural_network.optimizers import *

def main():
    # REGRESSION
    nn = NeuralNetwork(
        layers=[
            Dense(12, activation=PReLU(),  weight_decay=0.001),
            Dense(18, activation=Tanh(),   weight_decay=0.001),
            Dropout(dropout_rate=0.5),
            Dense(12, activation=Swish(),  weight_decay=0.001),
            Dense(3,  activation=Linear(), weight_decay=0.001)
        ]
    )
    nn.build(input_size=4)
    trainer = Trainer(nn, loss_function=Huber(delta=1.0), optimizer=Adam(learn_rate=0.02))

    # MULTILABEL
    # nn = NeuralNetwork(
    #     layers=[
    #         Dense(10, activation=Tanh(),    weight_decay=0.001),
    #         Dense(16, activation=Tanh(),    weight_decay=0.001),
    #         Dropout(dropout_rate=0.4),
    #         Dense(12, activation=Tanh(),    weight_decay=0.001),
    #         Dense(3,  activation=Sigmoid(), weight_decay=0.001)
    #     ]
    # )
    # nn.build(input_size=4)
    # trainer = Trainer(nn, loss_function=BCE(), optimizer=Adam(learn_rate=0.02))

    # MULTICLASS
    # nn = NeuralNetwork(
    #     layers=[
    #         Dense(12, activation=PReLU(),   weight_decay=0.001),
    #         Dense(16, activation=Tanh(),    weight_decay=0.001),
    #         Dropout(dropout_rate=0.4, batch_wise=True),
    #         Dense(12, activation=Swish(),   weight_decay=0.001),
    #         Dropout(dropout_rate=0.4),
    #         Dense(3,  activation=Softmax(), weight_decay=0.001)
    #     ]
    # )
    # nn.build(input_size=6)
    # trainer = Trainer(nn, loss_function=CCE(), optimizer=Adam(learn_rate=0.02))

    data_generator = DataGenerator()
    problem_type = "regression"

    input_train_list, output_train_list = data_generator.generate(40000, problem_type)
    input_test_list,  output_test_list  = data_generator.generate(20000, problem_type)
    showcase_i,       showcase_o        = data_generator.generate(16, problem_type)

    trainer.train(input_list=input_train_list,
             output_list=output_train_list,
             epoch=2000,
             batch_size=32)
    # nn.utils.inspect_weights_and_biases()

    nn.metrics.check_accuracy(test_input=input_test_list, test_output=output_test_list)

    nn.metrics.compare_predictions(input=showcase_i, output=showcase_o)

    input("Press any key to exit.")

if __name__ == "__main__":
    main()