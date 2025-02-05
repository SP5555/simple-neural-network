from data_generator import DataGenerator
from neural_network import NeuralNetwork
from neural_network.layers import *
from neural_network.optimizers import *

def main():
    nn = NeuralNetwork(
        layers=[
            DropoutLayer    (4, 12, "prelu",   dropout_rate=0.2,                  weight_decay=0.001),
            DropoutLayer    (12, 16, "tanh",   dropout_rate=0.2, batch_wise=True, weight_decay=0.001),
            DropoutLayer    (16, 12, "swish",  dropout_rate=0.2,                  weight_decay=0.001),
            DenseLayer      (12, 3, "softmax",                                    weight_decay=0.001)
        ],
        # optimizer=AdaGrad(learn_rate=0.01),
        optimizer=Momentum(learn_rate=0.02, momentum=0.75),
        loss_function="CCE"
    )

    data_generator = DataGenerator()
    problem_type = "multiclass"

    input_train_list, output_train_list = data_generator.generate(40000, problem_type)
    input_test_list, output_test_list = data_generator.generate(20000, problem_type)
    showcase_i, showcase_o = data_generator.generate(16, problem_type)

    nn.train(input_list=input_train_list,
             output_list=output_train_list,
             epoch=2000,
             batch_size=32)
    # nn.utils.inspect_weights_and_biases()

    nn.metrics.check_accuracy(test_input=input_test_list, test_output=output_test_list)

    nn.metrics.compare_predictions(input=showcase_i, output=showcase_o)

    input("Press any key to exit.")

if __name__ == "__main__":
    main()