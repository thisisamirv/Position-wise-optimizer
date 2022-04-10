#!/usr/local/bin/python3

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import time

# Cat classification dataset
data = 1
data_name = "Cat vs. non-cat classification"

import h5py
train_dataset = h5py.File('CatClassification.h5', "r")
X = np.array(train_dataset["train_set_x"][:])
Y = np.array(train_dataset["train_set_y"][:])
Y = Y.reshape((1, Y.shape[0]))
N = X.shape[0]
X_flatten = X.reshape(X.shape[0], -1).T
X = X_flatten / 255.
out_dim = 1

cost_lim_up = 0.7
cost_lim_down = 0.1
print_num = 100

# Model parameters
layer_dims = [X.shape[0], 20, 7, 5, out_dim]
learning_rate = 0.0075
epochs = 350
epsilon = 1e-5

def cost_function(Y, A4, N, epsilon):
    cost = (-1 / N) * np.sum(np.multiply(Y, np.log(A4 + epsilon)) + np.multiply(1 - Y, np.log(1 - A4 + epsilon)))
    return cost

# Classification layer
if data == 1:
    def class_layer(Z4):
        A4 = 1 / (1 + np.exp(-Z4))
        return A4
else:
    def class_layer(Z4):
        Z4_exp = np.exp(Z4)
        Z4_sum = np.sum(Z4_exp, axis = 0, keepdims = True)
        A4 = np.divide(Z4_exp, Z4_sum)
        return A4

# Print dataset info
print("====================================")
print("Dataset: " + data_name)
print("Number of training examples: " + str(N))
print("Y shape: " + str(Y.shape))
print("X shape: " + str(X.shape))
print("====================================")
print()

# Initialize parameters
W1 = np.random.randn(layer_dims[1], layer_dims[0]) / np.sqrt(layer_dims[0])
b1 = np.zeros((layer_dims[1], 1))
W2 = np.random.randn(layer_dims[2], layer_dims[1]) / np.sqrt(layer_dims[1])
b2 = np.zeros((layer_dims[2], 1))
W3 = np.random.randn(layer_dims[3], layer_dims[2]) / np.sqrt(layer_dims[2])
b3 = np.zeros((layer_dims[3], 1))
W4 = np.random.randn(layer_dims[4], layer_dims[3]) / np.sqrt(layer_dims[3])
b4 = np.zeros((layer_dims[4], 1))

W1_GD = np.copy(W1)
W2_GD = np.copy(W2)
W3_GD = np.copy(W3)
W4_GD = np.copy(W4)
b1_GD = np.copy(b1)
b2_GD = np.copy(b2)
b3_GD = np.copy(b3)
b4_GD = np.copy(b4)

W1_LW = np.copy(W1)
W2_LW = np.copy(W2)
W3_LW = np.copy(W3)
W4_LW = np.copy(W4)
b1_LW = np.copy(b1)
b2_LW = np.copy(b2)
b3_LW = np.copy(b3)
b4_LW = np.copy(b4)

cost_list_GD = []
cost_list_LW = []

# Geadient descent optimizer
t0 = time.process_time()

for i in range(epochs):
    Z1_GD = np.dot(W1_GD, X) + b1_GD
    A1_GD = np.maximum(0, Z1_GD)
    Z2_GD = np.dot(W2_GD, A1_GD) + b2_GD
    A2_GD = np.maximum(0, Z2_GD)
    Z3_GD = np.dot(W3_GD, A2_GD) + b3_GD
    A3_GD = np.maximum(0, Z3_GD)
    Z4_GD = np.dot(W4_GD, A3_GD) + b4_GD
    A4_GD = class_layer(Z4_GD)

    dZ4_GD = A4_GD - Y
    dW4_GD = np.dot(dZ4_GD, A3_GD.T) * (1. / A3_GD.shape[1])
    db4_GD = np.sum(dZ4_GD, axis=1, keepdims=True) * (1. / A3_GD.shape[1])

    dA3_GD = np.dot(W4_GD.T, dZ4_GD)
    dZ3_GD = np.array(dA3_GD, copy=True)
    dZ3_GD[Z3_GD <= 0] = 0
    dW3_GD = np.dot(dZ3_GD, A2_GD.T) * (1. / A2_GD.shape[1])
    db3_GD = np.sum(dZ3_GD, axis=1, keepdims=True) * (1. / A2_GD.shape[1])

    dA2_GD = np.dot(W3_GD.T, dZ3_GD)
    dZ2_GD = np.array(dA2_GD, copy=True)
    dZ2_GD[Z2_GD <= 0] = 0
    dW2_GD = np.dot(dZ2_GD, A1_GD.T) * (1. / A1_GD.shape[1])
    db2_GD = np.sum(dZ2_GD, axis=1, keepdims=True) * (1. / A1_GD.shape[1])

    dA1_GD = np.dot(W2_GD.T, dZ2_GD)
    dZ1_GD = np.array(dA1_GD, copy=True)
    dZ1_GD[Z1_GD <= 0] = 0
    dW1_GD = np.dot(dZ1_GD, X.T) * (1. / X.shape[1])
    db1_GD = np.sum(dZ1_GD, axis=1, keepdims=True) * (1. / X.shape[1])

    W1_GD = W1_GD - learning_rate * dW1_GD
    b1_GD = b1_GD - learning_rate * db1_GD
    W2_GD = W2_GD - learning_rate * dW2_GD
    b2_GD = b2_GD - learning_rate * db2_GD
    W3_GD = W3_GD - learning_rate * dW3_GD
    b3_GD = b3_GD - learning_rate * db3_GD
    W4_GD = W4_GD - learning_rate * dW4_GD
    b4_GD = b4_GD - learning_rate * db4_GD

    cost_GD = cost_function(Y, A4_GD, N, epsilon)
    cost_GD = np.squeeze(cost_GD)
    cost_list_GD.append(cost_GD)

    if i % print_num == 0:
        print("Cost for gradient descent optimizer after epoch {}: {}".format(i, cost_GD))
    elif cost_GD < cost_lim_down:
        last_i_GD = i
        print("Cost for gradient descent optimizer after epoch {}: {}".format(i, cost_GD))
        break
    else:
        continue

t1 = time.process_time() - t0
print("Time elapsed for gradient descent optimizer: {} seconds".format(t1))
print("====================================")
print()
