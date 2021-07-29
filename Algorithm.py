# Import packages, import and reshape data, and start and check some values
import numpy as np
import h5py

layer_dims = [12288, 20, 7, 5, 1]
L = len(layer_dims)
learning_rate = 0.0075

train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
train_x = np.array(train_dataset["train_set_x"][:])
train_y = np.array(train_dataset["train_set_y"][:])
test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
test_x = np.array(test_dataset["test_set_x"][:])
test_y = np.array(test_dataset["test_set_y"][:])
classes = np.array(test_dataset["list_classes"][:])
train_y = train_y.reshape((1, train_y.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))

m_train = train_x.shape[0]
num_px = train_x.shape[1]
num_px_test = test_x.shape[1]
m_test = test_x.shape[0]
m_train_y = train_y.shape[1]
m_test_y = test_y.shape[1]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x shape before reshape: " + str(train_x.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x shape before reshape: " + str(test_x.shape))
print("test_y shape: " + str(test_y.shape))

train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

X = train_x
Y = train_y

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
print("layers dimensions: " + str(L))

# Initialize parameters
W1 = np.random.randn(layer_dims[1], layer_dims[0]) / np.sqrt(layer_dims[0])
b1 = np.zeros((layer_dims[1], 1))
W2 = np.random.randn(layer_dims[2], layer_dims[1]) / np.sqrt(layer_dims[1])
b2 = np.zeros((layer_dims[2], 1))
W3 = np.random.randn(layer_dims[3], layer_dims[2]) / np.sqrt(layer_dims[2])
b3 = np.zeros((layer_dims[3], 1))
W4 = np.random.randn(layer_dims[4], layer_dims[3]) / np.sqrt(layer_dims[3])
b4 = np.zeros((layer_dims[4], 1))

# Network
for i in range(0, 500):
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0, Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = np.maximum(0, Z3)
    Z4 = np.dot(W4, A3) + b4
    A4 = 1 / (1 + np.exp(-Z4))

    dZ4 = A4 - Y
    dW4 = np.dot(dZ4, A3.T) * (1. / A3.shape[1])
    db4 = np.sum(dZ4, axis=1, keepdims=True) * (1. / A3.shape[1])

    dA3 = np.dot(W4.T, dZ4)
    s3 = 1 / (1 + np.exp(-Z3))
    sb3 = s3 * (s3 - 1)
    dZ3 = dA3 * sb3
    dW3 = np.dot(dZ3, A2.T) * (1. / A2.shape[1])
    db3 = np.sum(dZ3, axis=1, keepdims=True) * (1. / A2.shape[1])

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.array(dA2, copy=True)
    dZ2[Z2 <= 0] = 0
    dW2 = np.dot(dZ2, A1.T) * (1. / A1.shape[1])
    db2 = np.sum(dZ2, axis=1, keepdims=True) * (1. / A1.shape[1])

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.array(dA1, copy=True)
    dZ1[Z1 <= 0] = 0
    dW1 = np.dot(dZ1, X.T) * (1. / X.shape[1])
    db1 = np.sum(dZ1, axis=1, keepdims=True) * (1. / X.shape[1])

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    W4 = W4 - learning_rate * dW4
    b4 = b4 - learning_rate * db4
    if i % 100 == 0 or i == 500 - 1:
        cost = (-1 / m_train_y) * np.sum(np.multiply(train_y, np.log(A4)) + np.multiply(1 - train_y, np.log(1 - A4)))
        cost = np.squeeze(cost)
        print("Cost after iteration {}: {}".format(i, cost))

# Test
Z1_test = np.dot(W1, test_x) + b1
A1_test = np.maximum(0, Z1_test)
Z2_test = np.dot(W2, A1_test) + b2
A2_test = np.maximum(0, Z2_test)
Z3_test = np.dot(W3, A2_test) + b3
A3_test = np.maximum(0, Z3_test)
Z4_test = np.dot(W4, A3_test) + b4
A4_test = 1 / (1 + np.exp(-Z4_test))

cost_test = (-1 / m_test_y) * np.sum(
    np.multiply(test_y, np.log(A4_test)) + np.multiply(1 - test_y, np.log(1 - A4_test)))
cost_test = np.squeeze(cost_test)
print(cost_test)
