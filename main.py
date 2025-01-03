from NeuralNetwork import NeuralNetwork

import random

def data_generator(n: int):
    input_list: list = []
    output_list: list = []

    for _ in range(n):
        i1: float = random.uniform(-6, 6)
        i2: float = random.uniform(-6, 6)
        o1: float = 0.0

        # Make your own Data Here
        if i1*i1 - 5*i2 < 4:
            o1 = 1.0

        # noise for inputs
        i1 += random.uniform(-0.2, 0.2)
        i2 += random.uniform(-0.2, 0.2)

        input_list.append([i1, i2])
        output_list.append([o1])
    return input_list, output_list

def main():
    nn = NeuralNetwork(layers=[2, 8, 6, 1],
                       activation_hidden='leaky_relu',
                       activation_output='sigmoid',
                       learn_rate=0.08,
                       lambda_parem=0.003,
                       momentum=0.75)
    
    input_train_list, output_train_list = data_generator(40000)
    input_test_list, output_test_list = data_generator(40000)
    showcase_i, showcase_o = data_generator(16)

    # print("Training Started")
    nn.train(input_list=input_train_list,
             output_list=output_train_list,
             epoch=2000,
             batch_size=192)
    # nn.inspect_weights_and_biases()

    nn.check_accuracy_classification(test_input=input_test_list, test_output=output_test_list)

    print(" Expected | Predicted | Input Data")
    output = nn.forward_batch(showcase_i)
    for i in range(16):
        print(f'{showcase_o[i][0]:>9.4f} | {output[i][0]:>9.4f} | '
              f'{showcase_i[i][0]:.3f} {showcase_i[i][1]:.3f}')

if __name__ == "__main__":
    main()