# Imported Modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Defined Functions
def SplitInputsAndOutput(input_output_vector):
    """
    Purpose: Split the mixed inputs and output vector in each alone
    :param input_output_vector: the mixed vector [numpy]
    :return: inputs and output vector
    """
    # Defined Variables
    inputs_vector = np.zeros([len(input_output_vector), 2])     # two as 2 features only needed
    output_vector = np.zeros(len(input_output_vector))

    # then make Splitting Process
    for i in range(len(input_output_vector)):
        inputs_vector[i] = input_output_vector[i][:2]
        output_vector[i] = input_output_vector[i][2]

    # then return The wanted vectors
    return inputs_vector, output_vector

def CalulateNetValue(input, weight, bias):
    """
    Purpose: Calculate Net value
    :param inputs: the input vector of selected 2 features
    :param weights: the first tuning parameter
    :param bias: the second tuning parameter
    :return: net value
    """
    return input.dot(weight.transpose()) + bias

def Signum(netValue):
    """
    Purpose: apply Signum Activation function as net>0 return 1 else return -1
    :param netValue: the net values for each input [vector]
    :return: the result values [vector]
    """
    # Here I used where to process each value in vector
    prediction = np.where(netValue > 0, 1, -1)
    return prediction

def MakePrediction(input, weight, bias):
    """
    purpose: Predict the output value of inputs
    :param input: the data you want to predict y for [train/test]
    :param weight: the coefficients [tuning parameter]
    :param bias: the intersection [tuning parameter]
    :return: the prediction 1 or -1
    """
    # Calculate Net value and Apply Signum Activation Function
    net_value = CalulateNetValue(input, weight, bias)
    prediction = Signum(net_value)

    # return predicted value
    return prediction

def AdalineLearningAlgorithm(input, actual_output, weight, bias, epochs, thresh, alpha):
    """
    Purpose: Apply Adaline Algorithm
    :param input: the input features [2D vector]
    :param actual_output: the actual output of inputs [1D vector]
    :param weight: the coefficient of input [1D vector]
    :param bias: the intersection of linear equation
    :param epochs: number of iterations
    :param thresh: stopping Criteria of MSE
    :param alpha: learning rate
    :return: updated weigh and bias
    """
    # Start of Algorithm
    for epoch in range(epochs):
        # Calculate NetValue for Selected inputs
        predicted_output = CalulateNetValue(input, weight, bias)

        # Update Weights and Minimization of Error Process
        for i in range(len(predicted_output)):
            error = actual_output[i] - predicted_output[i]
            weight = weight + (alpha * error * input[i])
            if bias != 0:  # Here only check depends on user choice
                bias = bias + (alpha * error)

        # Calculate MSE For Last update of weight
        MSE = CalculateMSE(input, actual_output, weight, bias)
        if MSE < thresh:
            # Then Return The Weight and Bias
            print("--> MSE STATUS: [GOOD NEWS] the current MSE is less than threshold .. learning gonna stop now")
            return weight, bias

    # if That point reached that means The Returned MSE doesn't Match Stopping Criteria
    print("--> MSE STATUS: [BAD NEWS] for all {} epoch there is no MSE is less than threshold".format(epochs))
    return weight, bias

def CalculateMSE(input, actual_output, weight, bias):
    # Calculate Predicted Value for Last weights
    square_error = list()
    prediction = CalulateNetValue(input, weight, bias)

    # Calculate square of error for all data
    for i in range(len(actual_output)):
        error = np.square(actual_output[i] - prediction[i])
        square_error.append(error)

    # Then Calculate Mean square of error
    first_part = 1/len(square_error)
    second_part = sum(square_error)/2
    MSE = first_part * second_part

    # Then Return it
    return MSE

def GetPointsForLine(point_a_x, point_a_y, weight, bias):
    """
    purpose: Get The two points to draw line from equation
             X1*W1 + X2*W2 + bias = 0
    :param point_a_x: x position of the first point
    :param point_a_y: y position of the first point
    :param weight: Coefficients of Equation [w1, w2]
    :param bias: the intersection
    :return: the two points
    """
    # Here Calculate The other point coordinates
    point_b_x = (-weight[0]*point_a_x - bias) / weight[1]
    point_b_y = (-weight[0]*point_a_y - bias) / weight[1]

    # Merge point together in single vector
    point_a = np.array([point_a_x, point_a_y])
    point_b = np.array([point_b_x, point_b_y])

    # return two points
    return point_a, point_b

def DrawLine(data, features, start_point, end_point, actual_output):
    """
    Purpose: Scatter Selected two features and draw fitted line
    :param data: the Whole data
    :param features: only needed features from data
    :param start_point: x coordinate for fitted line
    :param end_point: y coordinate for fitted line
    :param actual_output: the output of data for grouping similar output with same color
    :return: Void
    """
    plt.figure("Trained Features Figure")
    plt.scatter(x=data[features[0]], y=data[features[1]], c=actual_output)
    plt.plot(start_point, end_point)
    plt.show()

def CreateConfuionMatrix(actual_output, prediction):
    """
    Purpose: Create Confusion Matrix and Print output status
    :param actual_output: the real values of tested data
    :param prediction: the calculated output of trained parameters
    :return: accuracy
    """
    # Defined Variables
    columns = ['actual', 'prediction', 'status']
    row = list()
    test_status = pd.DataFrame(columns=columns)
    confusion_matrix = np.zeros([2, 2])

    # Creating Confusion Matrix Process and Get status of output
    for i in range(len(prediction)):
        if actual_output[i] == 1:
            if prediction[i] == 1:
                confusion_matrix[0][0] += 1
                row.append([actual_output[i], prediction[i], 'Matching'])
            else:
                confusion_matrix[0][1] += 1
                row.append([actual_output[i], prediction[i], 'Mismatching'])

        elif actual_output[i] == -1:
            if prediction[i] == -1:
                confusion_matrix[1][1] += 1
                row.append([actual_output[i], prediction[i], 'Matching'])
            else:
                confusion_matrix[1][0] += 1
                row.append([actual_output[i], prediction[i], 'Mismatching'])

    # Then Print status of each output
    rows = pd.DataFrame(row, columns=columns)
    test_status = test_status.append(rows, ignore_index=True)
    print("--> The Test is Now Running ...")
    print(test_status)

    # Calculate accuracy by [sum of diagonal / total sum]
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # Show Confusion Matrix
    print("--> Confusion Matrix:")
    print(confusion_matrix)

    # return the accuracy
    return accuracy
