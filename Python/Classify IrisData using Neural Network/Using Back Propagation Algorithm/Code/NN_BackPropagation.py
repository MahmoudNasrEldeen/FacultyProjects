# Imported Modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def MakeHotEncoderForOutput(dataset, header):
    # The process of making Encoder
    headers = ['C1', 'C2', 'C3', 'X1', 'X2', 'X3', 'X4']
    ct = ColumnTransformer([(header, OneHotEncoder(),
                           [dataset.columns.get_loc(header)])],
                           remainder='passthrough')
    dataset = ct.fit_transform(dataset)
    dataset = pd.DataFrame(dataset, columns=headers)

    # the dataset after making Encoder
    return dataset

def inputDataPreprocessing(dataset):
    # Get Classes Values then Shuffle it
    class_a, class_b, class_c = dataset[:50], dataset[50:100], dataset[100:]
    class_a, class_b, class_c = class_a.sample(frac=1), class_b.sample(frac=1), class_c.sample(frac=1)

    # Then Split The data into 30 Train and 20 Test
    class_a_train, class_a_test = class_a[:30], class_a[30:]
    class_b_train, class_b_test = class_b[:30], class_b[30:]
    class_c_train, class_c_test = class_c[:30], class_c[30:]

    # Merge trained data together and tested data together
    trained_data = pd.concat([class_a_train, class_b_train, class_c_train])
    tested_data = pd.concat([class_a_test, class_b_test, class_c_test])

    # Then return both
    return trained_data, tested_data

def SplitInputsAndOutput(dataset):
    # Initialize matrices with suitable shapes
    inputs_matrix = np.zeros([len(dataset), 5])  # 4 inputs + 1 bias
    output_matrix = np.zeros([len(dataset), 3])

    # Make Split Process
    for i in range(len(dataset)):
        output_matrix[i] = dataset[i][0:3]
        inputs_matrix[i] = dataset[i][3:]

    # Then return both
    return inputs_matrix, output_matrix

def GetSuitableWeight(hidden_layers, neurons_number):
    # Create weight list that carry matrices
    weights_list = list()

    # According to number of layers create weight matrices
    weights_list.append(np.random.randn(5, neurons_number[0]))
    for i in range(hidden_layers - 1):
        rows = neurons_number[i]
        columns = neurons_number[i + 1]
        weights_list.append(np.random.randn(rows, columns))
    weights_list.append(np.random.randn(neurons_number[-1], 3))

    # Return The list that included matrices of weights
    return weights_list

def WorkwithBias(data, bias_decision):
    if bias_decision == 0:
        # Then Add bias to Trained Data with value = 0
        bias_vector = [0 for _ in range(len(data))]
    else:
        # Then Add bias to Trained Data with value = 1
        bias_vector = [1 for _ in range(len(data))]

    # Then add it to data input and return whole trained/tested data
    data.insert(loc=3, column='bias', value=bias_vector)
    return data

def workWithBias_Hidden(biasDes, neuronsVal):
    # if user don't need to use bias then
    # let neuron that represents it be equal zero
    if biasDes == 0:
        for neuron in range(len(neuronsVal) - 1):
            neuronsVal[neuron][0] = 0

    # But if user need to use bias then
    # let neuron that represents it be equal one
    else:
        for neuron in range(len(neuronsVal) - 1):
            neuronsVal[neuron][0] = 1

def CreateNeuronsMat(neurons):
    # Define the used matrix (list of matrices)
    neurons_Mat = list()
    for neuron in neurons:
        neurons_Mat.append(np.zeros(neuron))
    neurons_Mat.append(np.zeros(3))  # as output consists of 3 neurons

    # Then return it
    return neurons_Mat

def CalculateNetValue(single_input, weight):
    # Calculate and return net value for given input
    return single_input.dot(weight)

def Sigmoid(netVal):
    # Calculate and return Sigmoid function for net value
    return 1 / (1 + np.exp(-netVal))

def HyperboicTangent(netVal):
    # Calculate and return Hyperbolic tangent function for net value
    #sinh = np.exp(netVal) - np.exp(-netVal)
    #cosh = np.exp(netVal) + np.exp(-netVal)
    #return sinh / cosh
    return np.tanh(netVal)

def GetExpectedNeuronValue(inpVal, weight, function_to_use):
    # Calculate net value and apply activation function based on user choice
    netVal = CalculateNetValue(inpVal, weight)
    if function_to_use == "Sigmoid":
        expectVal = Sigmoid(netVal)
    else:
        expectVal = HyperboicTangent(netVal)

    # Then return Calculated value
    return expectVal

def GetNetworkOutput(predicted_outputs):
    # The process of making the output of the network understandable
    # by making the max value in output layer be 1 and others be 0s
    for output in predicted_outputs:
        max_index = np.argmax(output)
        for index in range(len(output)):
            if index == max_index:
                output[index] = 1
            else:
                output[index] = 0

    # Then return it
    return predicted_outputs

def GetWeightShape(weights, index):
    # Get shape of each weight matrix for update step
    x_shape, y_shape = weights[index].shape
    return x_shape, y_shape

def STEP_A_Feedforward(single_input, weights, activ_function, hidden, neurons_output, useBias):
    # For all hidden layers + output layer
    for layer in range(0, hidden + 1):
        if layer == 0:
            # Get Neurons Value between input and first Hidden Layer
            neurons_output[layer] = GetExpectedNeuronValue(single_input, weights[layer], activ_function)
            continue

        # Then Get Neurons Value for remaining Hidden layers included output layer
        neurons_output[layer] = GetExpectedNeuronValue(neurons_output[layer - 1], weights[layer], activ_function)

    # then let first neuron works as bias
    workWithBias_Hidden(biasDes=useBias, neuronsVal=neurons_output)

    # Return The list that carry each neuron output
    return neurons_output

def STEP_B_Backpropagate(actual_output, predicted_output, weights, neurons_values, local_gradient, hidden):
    # Walk layer per layer till the input layer (from back to front)
    for layer in range(hidden, -1, -1):
        activ_fun_drev = neurons_values[layer] * (1 - neurons_values[layer])

        # for Output Layer
        if layer == hidden:
            error = actual_output - predicted_output
            local_gradient[layer] = error * activ_fun_drev
            continue

        # For hidden Layers
        sumPart = local_gradient[layer + 1].dot(weights[layer + 1].transpose())
        local_gradient[layer] = activ_fun_drev * sumPart

    # Return The Local Minimum for all neurons
    return local_gradient

def STEP_C_UpdateWeights(sample_inp, weights, hidden_layers, local_gradient, neurons_out, eta):
    for layer in range(hidden_layers + 1):
        # update the weights of first layer as it depends on input from data
        if layer == 0:
            x_shape, y_shape = GetWeightShape(weights=weights, index=layer)
            input_x_local = sample_inp.reshape(x_shape, 1).dot(local_gradient[layer].reshape(1, y_shape))
            weights[layer] = weights[layer] + (eta * input_x_local)
            continue

        # then update other layers(hidden+output) as it there depends on input from previous neurons
        x_shape, y_shape = GetWeightShape(weights=weights, index=layer)
        input_x_local = neurons_out[layer - 1].reshape(x_shape, 1).dot(local_gradient[layer].reshape(1, y_shape))
        weights[layer] = weights[layer] + (eta * input_x_local)

    # Then return optimal weights
    return weights

def BackPropagationAlgorithm(inputs, output, weight, activation, epochs, eta, layers, neurons, useBias):
    # Create List carry vector for each neuron output and Local Minimum per layer
    neurons_output = CreateNeuronsMat(neurons)
    localGrad = CreateNeuronsMat(neurons)

    # Learning Start from here
    for epoch in range(epochs):
        for i in range(len(inputs)):
            # STEP 1: Apply Feedforward
            neurons_output = STEP_A_Feedforward(single_input=inputs[i],
                                                weights=weight,
                                                activ_function=activation,
                                                hidden=layers,
                                                neurons_output=neurons_output,
                                                useBias=useBias)

            # STEP 2: Apply Backpropagation
            localGrad = STEP_B_Backpropagate(actual_output=output[i],
                                             predicted_output=neurons_output[-1],
                                             weights=weight,
                                             neurons_values=neurons_output,
                                             local_gradient=localGrad,
                                             hidden=layers)

            # STEP 3: Update Weights
            weight = STEP_C_UpdateWeights(sample_inp=inputs[i],
                                          weights=weight,
                                          hidden_layers=layers,
                                          local_gradient=localGrad,
                                          neurons_out=neurons_output,
                                          eta=eta)

    # After learning finished return optimal weights reached
    return weight

def ApplyTestScenario(inputs, weights, activ_function, hidden, neurons_output, useBias):
    # test data by applying feedforward step using calculated optimal weights
    predicted_output = list()
    for i in range(len(inputs)):
        total_neurons_values = STEP_A_Feedforward(single_input=inputs[i],
                                                  weights=weights,
                                                  activ_function=activ_function,
                                                  hidden=hidden,
                                                  neurons_output=neurons_output,
                                                  useBias=useBias)
        predicted_output.append(total_neurons_values[-1])

    # Then Get The output for all tested data and return it
    network_output = GetNetworkOutput(predicted_output)
    return network_output

def CreateConfusionMatrix(output_test, network_output):
    # Defined Variables
    confusion_matrix = np.zeros([3, 3])
    flowers = ['setosa', 'versicolor', 'virginica']
    columns = ['Class ID', 'Predicted', 'Actual', 'Status']
    row = list()
    test_status = pd.DataFrame(columns=columns)

    for index in range(len(output_test)):

        # Building the Confusion Matrix
        actual_index = np.argmax(output_test[index])
        predicted_index = np.argmax(network_output[index])
        confusion_matrix[actual_index][predicted_index] += 1

        # Here for visualization part
        classID = predicted_index + 1
        className = flowers[predicted_index]
        actualClass = flowers[actual_index]
        if actual_index == predicted_index:
            status = "Matching"
        else:
            status = "Mismatching"
        row.append([classID, className, actualClass, status])

    # Then Print status of each output
    rows = pd.DataFrame(row, columns=columns)
    test_status = test_status.append(rows, ignore_index=True)
    print("------------------- Test Result --------------------")
    print(test_status)
    print()

    # print Confusion Matrix
    classes = ['C1', 'C2', 'C3']
    print("---------------- Confusion Matrix ------------------")
    matrix = pd.DataFrame(confusion_matrix, columns=classes, index=classes)
    print(matrix)
    print()

    # Then print accuracy by [sum of diagonal / total sum] and for each class Too
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    percentage = len(output_test) / len(classes)
    print("------------------- Accuracy -----------------------")
    for i in range(len(classes)):
        print("Accuracy for Class {}: {}".format(i+1, confusion_matrix[i][i]/percentage))
    print("Accuracy for whole network: {}".format(accuracy))
