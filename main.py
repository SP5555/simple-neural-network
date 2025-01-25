from neural_network import NeuralNetwork
from data_generator import DataGenerator

def main():
    nn = NeuralNetwork(
        layers=[4, 12, 12, 3],
        activation=["leaky_relu", "sigmoid", "softmax"],
        loss_function="CCE",
        learn_rate=0.02,
        lambda_parem=0.003,
        momentum=0.75
    )

    data_generator = DataGenerator()
    problem_type = "multiclass"

    input_train_list, output_train_list = data_generator.generate(40000, problem_type)
    input_test_list, output_test_list = data_generator.generate(20000, problem_type)
    showcase_i, showcase_o = data_generator.generate(16, problem_type)

    nn.train(input_list=input_train_list,
             output_list=output_train_list,
             epoch=2000,
             batch_size=64)
    # nn.inspect_weights_and_biases()

    nn.metrics.check_accuracy_classification(test_input=input_test_list, test_output=output_test_list)

    nn.metrics.compare_predictions(input=showcase_i, output=showcase_o)

if __name__ == "__main__":
    main()