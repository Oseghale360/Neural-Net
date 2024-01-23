# Oseghale, Emmanuel Emakhue
# 1001_908_482
# 2023_09_25
# Assignment_01_01

import numpy as np


def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2):
    # This function creates and trains a multi-layer neural Network
    # X_train: Array of input for training [input_dimensions,nof_train_samples]

    # Y_train: Array of desired outputs for training samples [output_dimensions,nof_train_samples]
    # X_test: Array of input for testing [input_dimensions,nof_test_samples]
    # Y_test: Array of desired outputs for test samples [output_dimensions,nof_test_samples]
    # layers: array of integers representing number of nodes in each layer
    # alpha: learning rate
    # epochs: number of epochs for training.
    # h: step size
    # seed: random number generator seed for initializing the weights.
    # return: This function should return a list containing 3 elements:
    # The first element of the return list should be a list of weight matrices.
    # Each element of the list corresponds to the weight matrix of the corresponding layer.

    # The second element should be a one dimensional array of numbers
    # representing the average mse error after each epoch. Each error should
    # be calculated by using the X_test array while the network is frozen.
    # This means that the weights should not be adjusted while calculating the error.

    # The third element should be a two-dimensional array [output_dimensions,nof_test_samples]
    # representing the actual output of network when X_test is used as input.

    # Notes:
    # DO NOT use any other package other than numpy
    # Bias should be included in the weight matrix in the first column.
    # Assume that the activation functions for all the layers are sigmoid.
    # Use MSE to calculate error.
    # Use gradient descent for adjusting the weights.
    # use centered difference approximation to calculate partial derivatives.
    # (f(x + h)-f(x - h))/2*h
    # Reseed the random number generator when initializing weights for each layer.
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
    outputs = []
    weights = initWeights(X_train.shape[0], layers, Y_train.shape[0], seed)
    outputs.append(weights)
    outputs.append([])
    testOutputs = np.zeros((layers[-1], Y_test.shape[1]))
    mses = np.zeros(epochs)


    for epoch in range(epochs):
        for x in range(X_train.shape[1]):
            s_input = X_train[:, x]
            s_input = s_input.reshape(1, s_input.shape[0])
            output = calNetOutput(s_input, weights)

            weights = updateWeights(weights, s_input, Y_train[:, x], alpha, h)

        mse = np.zeros(Y_test.shape[1])
        for x in range(X_test.shape[1]):
            s_input = X_test[:, x]
            s_input = s_input.reshape(1, s_input.shape[0])
            output = calNetOutput(s_input, weights)
            testOutputs[:, x] = output
            mse[x] = calcMSE(Y_test[:, x], output)
        mses[epoch] = np.mean(mse)
    outputs[0] = weights
    outputs[1] = mses

    outputs.append(testOutputs)

    return outputs

def calcMSE(target, present):
    lent = len(target)
    res = 0
    for x in range(lent):
        res = res + ((target[x] - present[x]) ** 2)
    return res / lent


def updateWeights(weightsArr, inputData, target, alpha, step):
    newWeightsArr = []
    for w in range(len(weightsArr)):
        weights = weightsArr[w]
        if len(weights.shape) == 1:
            weights.reshape(weights.shape[0], 1)
        newWeights = np.zeros(weights.shape)
        rows, cols = weights.shape
        for row in range(rows):
            weights = weightsArr[w]
            for col in range(cols):
                orgWeight = weights[row, col]
                weights[row, col] = orgWeight + step
                weightsArr[w] = weights
                output1 = calNetOutput(inputData, weightsArr)
                mse1 = calcMSE(target, output1)

                weights[row, col] = orgWeight - step
                weightsArr[w] = weights
                output2 = calNetOutput(inputData, weightsArr)
                mse2 = calcMSE(target, output2)

                stepWei = (mse1 - mse2) / (2 * step)
                newWei = orgWeight - (alpha * stepWei)

                weights[row, col] = orgWeight

                newWeights[row, col] = newWei
        newWeightsArr.append(newWeights)
    return newWeightsArr






def initWeights(inSize, layers, outSize, seed):
    np.random.seed(seed)
    weights = []
    bias = []
    layers = [inSize, *layers]#, outSize
    netWeights = []
    for layer in range(len(layers)-1):
        np.random.seed(seed)
        weights = np.zeros((layers[layer+1], layers[layer]+1))
        for y in range(layers[layer+1]):
            weight = np.zeros(layers[layer]+1)
            for x in range(layers[layer]):
                weight[x] = np.random.randn()
            weight[x+1] = np.random.randn()
            weights[y, :] = weight
        netWeights.append(weights)
    return netWeights


def calLayerOutput(layerInput, layerWeights):
    out = np.dot(layerInput, layerWeights.T)
    return out[0]

def sigmoid(x):
    # This function calculates the sigmoid function
    # x: input
    # return: sigmoid(x)
    # Your code goes here
    return 1/(1+np.exp(-x))


def calNetOutput(xTrain, netWeights):
    Y = None
    for layer in range(len(netWeights)):
        inputDat = xTrain if Y is None else Y
        inputDat = np.append([1], inputDat)
        inputDat = inputDat.reshape(1, inputDat.shape[0])
        Y = calLayerOutput(inputDat, netWeights[layer])
        Y = sigmoid(Y)

    return Y