# Imported Modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Defined Functions
def TwoFeaturesScatter(feature_a, feature_b, label_a, label_b):
    """
    Purpose: Scatter Whole data with each flower type has its own color
    :param feature_a: the first feature you depend on classification e.g x1, x2, ...
    :param feature_b: the second feature you depend on classification e.g x1, x2, ...
    :param label_a: first feature name to display in figure
    :param label_b: second feature name to display in figure
    :return: void
    """
    # Define scatter info
    plt.figure("Flower Features Figure")
    plt.xlabel(label_a)
    plt.ylabel(label_b)

    # Divide features according to its flower
    setosa_feature_a = feature_a[:51]
    setosa_feature_b = feature_b[:51]
    versicolor_feature_a = feature_a[51:101]
    versicolor_feature_b = feature_b[51:101]
    virginica_feature_a = feature_a[101:]
    virginica_feature_b = feature_b[101:]

    # pass scatter data then show
    plt.scatter(x=setosa_feature_a, y=setosa_feature_b)
    plt.scatter(x=versicolor_feature_a, y=versicolor_feature_b)
    plt.scatter(x=virginica_feature_a, y=virginica_feature_b)
    plt.show()

def DrawIrisData(filename):
    """
    purpose: Plotting all possible Combination of features x1, x2, x3, x4
    :param filename: the file you want to read
    :return: void
    """
    dataset = pd.read_csv(filename)
    TwoFeaturesScatter(feature_a=dataset['X1'], feature_b=dataset['X2'], label_a='X1', label_b='X2')
    TwoFeaturesScatter(feature_a=dataset['X1'], feature_b=dataset['X3'], label_a='X1', label_b='X3')
    TwoFeaturesScatter(feature_a=dataset['X1'], feature_b=dataset['X4'], label_a='X1', label_b='X4')
    TwoFeaturesScatter(feature_a=dataset['X2'], feature_b=dataset['X3'], label_a='X2', label_b='X3')
    TwoFeaturesScatter(feature_a=dataset['X2'], feature_b=dataset['X4'], label_a='X2', label_b='X4')
    TwoFeaturesScatter(feature_a=dataset['X3'], feature_b=dataset['X4'], label_a='X3', label_b='X4')

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

def PerceptronLearningAlgorithm(input, actual_output, weight, bias, epochs, alpha):
    """
    Purpose: Apply Perceptron Algorithm
    :param input: the input features [2D vector]
    :param actual_output: the actual output of inputs [1D vector]
    :param weight: the coefficient of input [1D vector]
    :param bias: the intersection of linear equation
    :param epochs: number of iterations
    :param alpha: learning rate
    :return: updated weigh and bias
    """
    # Note: I here didn't prefer adding bias into weights vector for reason which is
    #       when user doesn't want use bias I don't need to work with bias also in updating
    #       but if we putting it in vector it will be updated e.g. b = 0 + alpha*loss
    #       bias = smallValue and that doesn't make sense as user doesn't want to use it
    for epoch in range(epochs):
        # Forward propagation case that calculate netValue and predict output
        predicted_output = MakePrediction(input, weight, bias)

        # Then here start case of back propagation as calculate loss if predicted
        # value not same of actual then update the weights for Minimizing Loss
        for i in range(len(actual_output)):
            if predicted_output[i] != actual_output[i]:
                loss = actual_output[i] - predicted_output[i]
                weight = weight + alpha*loss*input[i]
                if bias != 0:   # Here only check depends on user choice
                    bias = bias + alpha*loss

    # finally return the final updated weights and bias[if exists]
    return weight, bias

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
    print("The Test is Now Running ...")
    print(test_status)

    # Calculate accuracy by [sum of diagonal / total sum]
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # return the accuracy
    return accuracy


# Start Of Main
filename = "Dataset/IrisData.txt"
DrawIrisData(filename)
