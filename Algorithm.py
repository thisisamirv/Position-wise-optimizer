#!/usr/local/bin/python3

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import time

# Cat classification dataset
"""
import h5py
train_dataset = h5py.File('CatClassification.h5', "r")
train_x = np.array(train_dataset["train_set_x"][:])
train_y = np.array(train_dataset["train_set_y"][:])
train_y = train_y.reshape((1, train_y.shape[0]))
m = train_x.shape[0]
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
train_x = train_x_flatten / 255.
X = train_x
Y = train_y
out_dim = 1
data = 1

cost_lim_up = 0.7
cost_lim_down = 0.1
print_num = 100
"""

# MNIST dataset
import os
import requests
import gzip
import hashlib

path = 'MNIST'
os.makedirs(path, exist_ok=True)
def fetch(url):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()
X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
train_x = np.copy(X)
train_y = np.copy(Y)
train_y = train_y.reshape(-1)
train_y = (np.eye(10)[train_y]).T
train_x = train_x.reshape(train_x.shape[0], -1).T
train_y = train_y.astype(np.float32)
train_x = train_x.astype(np.float32)
m = train_x.shape[1]
X = train_x
Y = train_y
out_dim = 10
data = 2

cost_lim_up = 3.2496
cost_lim_down = 3.2494
print_num = 150


# CIFAR-10 dataset
"""
import os
import tarfile
from urllib.request import urlretrieve

path = 'CIFAR-10'
url = 'https://www.cs.toronto.edu/~kriz/'
tar = 'cifar-10-binary.tar.gz'
files = ['cifar-10-batches-bin/data_batch_1.bin',
         'cifar-10-batches-bin/data_batch_2.bin',
         'cifar-10-batches-bin/data_batch_3.bin',
         'cifar-10-batches-bin/data_batch_4.bin',
         'cifar-10-batches-bin/data_batch_5.bin',
         'cifar-10-batches-bin/test_batch.bin']
os.makedirs(path, exist_ok=True)
if tar not in os.listdir(path):
    urlretrieve(''.join((url, tar)), os.path.join(path, tar))
with tarfile.open(os.path.join(path, tar)) as tar_object:
    fsize = 10000 * (32 * 32 * 3) + 10000
    buffr = np.zeros(fsize * 6, dtype='uint8')
    members = [file for file in tar_object if file.name in files]
    members.sort(key=lambda member: member.name)
    for i, member in enumerate(members):
        f = tar_object.extractfile(member)
        buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')
labels = buffr[::3073]
pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
images = pixels.reshape(-1, 3072).astype('float32') / 255
train_images, test_images = images[:50000], images[50000:]
train_labels, test_labels = labels[:50000], labels[50000:]
n_rows = len(train_labels)
n_cols = train_labels.max() + 1
train_y = np.zeros((n_rows, n_cols), dtype='uint8')
train_y[np.arange(n_rows), train_labels] = 1
X = train_images.T
Y = train_y.T
m = X.shape[1]
out_dim = 10
data = 3

cost_lim_up = 3.2560
cost_lim_down = 3.230
print_num = 20
"""

# Model parameters
layer_dims = [X.shape[0], 20, 7, 5, out_dim]
L = len(layer_dims)
learning_rate = 0.0075
epochs = 15000
epsilon = 1e-5

def cost_function(Y, A4, m, epsilon):
    cost_n = (-1 / m) * np.sum(np.multiply(Y, np.log(A4 + epsilon)) + np.multiply(1 - Y, np.log(1 - A4 + epsilon)))
    return cost_n

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
print("Number of training examples: " + str(m))
print("train_y shape: " + str(Y.shape))
print("train_x's shape: " + str(X.shape))
print("layers dimensions: " + str(L))
print()

# Initialize parameters
np.random.seed(1)
W1 = np.random.randn(layer_dims[1], layer_dims[0]) / np.sqrt(layer_dims[0])
b1 = np.zeros((layer_dims[1], 1))
W2 = np.random.randn(layer_dims[2], layer_dims[1]) / np.sqrt(layer_dims[1])
b2 = np.zeros((layer_dims[2], 1))
W3 = np.random.randn(layer_dims[3], layer_dims[2]) / np.sqrt(layer_dims[2])
b3 = np.zeros((layer_dims[3], 1))
W4 = np.random.randn(layer_dims[4], layer_dims[3]) / np.sqrt(layer_dims[3])
b4 = np.zeros((layer_dims[4], 1))

W1n = np.copy(W1)
W2n = np.copy(W2)
W3n = np.copy(W3)
W4n = np.copy(W4)
b1n = np.copy(b1)
b2n = np.copy(b2)
b3n = np.copy(b3)
b4n = np.copy(b4)

W1j = np.copy(W1)
W2j = np.copy(W2)
W3j = np.copy(W3)
W4j = np.copy(W4)
b1j = np.copy(b1)
b2j = np.copy(b2)
b3j = np.copy(b3)
b4j = np.copy(b4)

cost_list_n = []
cost_list_j = []

# Conventional Backprop
t0 = time.process_time()

for i in range(epochs):
    Z1n = np.dot(W1n, X) + b1n
    A1n = np.maximum(0, Z1n)
    Z2n = np.dot(W2n, A1n) + b2n
    A2n = np.maximum(0, Z2n)
    Z3n = np.dot(W3n, A2n) + b3n
    A3n = np.maximum(0, Z3n)
    Z4n = np.dot(W4n, A3n) + b4n
    A4n = class_layer(Z4n)

    dZ4n = A4n - Y
    dW4n = np.dot(dZ4n, A3n.T) * (1. / A3n.shape[1])
    db4n = np.sum(dZ4n, axis=1, keepdims=True) * (1. / A3n.shape[1])

    dA3n = np.dot(W4n.T, dZ4n)
    dZ3n = np.array(dA3n, copy=True)
    dZ3n[Z3n <= 0] = 0
    dW3n = np.dot(dZ3n, A2n.T) * (1. / A2n.shape[1])
    db3n = np.sum(dZ3n, axis=1, keepdims=True) * (1. / A2n.shape[1])

    dA2n = np.dot(W3n.T, dZ3n)
    dZ2n = np.array(dA2n, copy=True)
    dZ2n[Z2n <= 0] = 0
    dW2n = np.dot(dZ2n, A1n.T) * (1. / A1n.shape[1])
    db2n = np.sum(dZ2n, axis=1, keepdims=True) * (1. / A1n.shape[1])

    dA1n = np.dot(W2n.T, dZ2n)
    dZ1n = np.array(dA1n, copy=True)
    dZ1n[Z1n <= 0] = 0
    dW1n = np.dot(dZ1n, X.T) * (1. / X.shape[1])
    db1n = np.sum(dZ1n, axis=1, keepdims=True) * (1. / X.shape[1])

    W1n = W1n - learning_rate * dW1n
    b1n = b1n - learning_rate * db1n
    W2n = W2n - learning_rate * dW2n
    b2n = b2n - learning_rate * db2n
    W3n = W3n - learning_rate * dW3n
    b3n = b3n - learning_rate * db3n
    W4n = W4n - learning_rate * dW4n
    b4n = b4n - learning_rate * db4n

    cost_n = cost_function(Y, A4n, m, epsilon)
    cost_n = np.squeeze(cost_n)
    cost_list_n.append(cost_n)

    if i % print_num == 0 or i == epochs - 1:
        print("Cost for conventional backprop after iteration {}: {}".format(i, cost_n))

    if cost_n < cost_lim_down:
        last_i_n = i
        print("Cost for conventional backprop after iteration {}: {}".format(i, cost_n))
        break
    else:
        continue

t1 = time.process_time() - t0
print("Time elapsed for conventional backprop: ", t1)
print()

# Optimized Backprop
t2 = time.process_time()

for i in range(epochs):
    Z1j = np.dot(W1j, X) + b1j
    A1j = np.maximum(0, Z1j)
    Z2j = np.dot(W2j, A1j) + b2j
    A2j = np.maximum(0, Z2j)
    Z3j = np.dot(W3j, A2j) + b3j
    A3j = np.maximum(0, Z3j)
    Z4j = np.dot(W4j, A3j) + b4j
    A4j = class_layer(Z4j)

    dZ4j = A4j - Y
    dW4j = np.dot(dZ4j, A3j.T) * (1. / A3j.shape[1])
    db4j = np.sum(dZ4j, axis=1, keepdims=True) * (1. / A3j.shape[1])
    W4j = W4j - learning_rate * dW4j
    b4j = b4j - learning_rate * db4j
    Z4j = np.dot(W4j, A3j) + b4j
    A4j = class_layer(Z4j)

    dZ4j = A4j - Y
    dW4j = np.dot(dZ4j, A3j.T) * (1. / A3j.shape[1])
    db4j = np.sum(dZ4j, axis=1, keepdims=True) * (1. / A3j.shape[1])
    dA3j = np.dot(W4j.T, dZ4j)
    dZ3j = np.array(dA3j, copy=True)
    dZ3j[Z3j <= 0] = 0
    dW3j = np.dot(dZ3j, A2j.T) * (1. / A2j.shape[1])
    db3j = np.sum(dZ3j, axis=1, keepdims=True) * (1. / A2j.shape[1])
    W3j = W3j - learning_rate * dW3j
    b3j = b3j - learning_rate * db3j
    W4j = W4j - learning_rate * dW4j
    b4j = b4j - learning_rate * db4j
    Z3j = np.dot(W3j, A2j) + b3j
    A3j = np.maximum(0, Z3j)
    Z4j = np.dot(W4j, A3j) + b4j
    A4j = class_layer(Z4j)

    dZ4j = A4j - Y
    dW4j = np.dot(dZ4j, A3j.T) * (1. / A3j.shape[1])
    db4j = np.sum(dZ4j, axis=1, keepdims=True) * (1. / A3j.shape[1])
    dA3j = np.dot(W4j.T, dZ4j)
    dZ3j = np.array(dA3j, copy=True)
    dZ3j[Z3j <= 0] = 0
    dW3j = np.dot(dZ3j, A2j.T) * (1. / A2j.shape[1])
    db3j = np.sum(dZ3j, axis=1, keepdims=True) * (1. / A2j.shape[1])
    dA2j = np.dot(W3j.T, dZ3j)
    dZ2j = np.array(dA2j, copy=True)
    dZ2j[Z2j <= 0] = 0
    dW2j = np.dot(dZ2j, A1j.T) * (1. / A1j.shape[1])
    db2j = np.sum(dZ2j, axis=1, keepdims=True) * (1. / A1j.shape[1])
    W2j = W2j - learning_rate * dW2j
    b2j = b2j - learning_rate * db2j
    W3j = W3j - learning_rate * dW3j
    b3j = b3j - learning_rate * db3j
    W4j = W4j - learning_rate * dW4j
    b4j = b4j - learning_rate * db4j
    Z2j = np.dot(W2j, A1j) + b2j
    A2j = np.maximum(0, Z2j)
    Z3j = np.dot(W3j, A2j) + b3j
    A3j = np.maximum(0, Z3j)
    Z4j = np.dot(W4j, A3j) + b4j
    A4j = class_layer(Z4j)

    dZ4j = A4j - Y
    dW4j = np.dot(dZ4j, A3j.T) * (1. / A3j.shape[1])
    db4j = np.sum(dZ4j, axis=1, keepdims=True) * (1. / A3j.shape[1])
    dA3j = np.dot(W4j.T, dZ4j)
    dZ3j = np.array(dA3j, copy=True)
    dZ3j[Z3j <= 0] = 0
    dW3j = np.dot(dZ3j, A2j.T) * (1. / A2j.shape[1])
    db3j = np.sum(dZ3j, axis=1, keepdims=True) * (1. / A2j.shape[1])
    dA2j = np.dot(W3j.T, dZ3j)
    dZ2j = np.array(dA2j, copy=True)
    dZ2j[Z2j <= 0] = 0
    dW2j = np.dot(dZ2j, A1j.T) * (1. / A1j.shape[1])
    db2j = np.sum(dZ2j, axis=1, keepdims=True) * (1. / A1j.shape[1])
    dA1j = np.dot(W2j.T, dZ2j)
    dZ1j = np.array(dA1j, copy=True)
    dZ1j[Z1j <= 0] = 0
    dW1j = np.dot(dZ1j, X.T) * (1. / X.shape[1])
    db1j = np.sum(dZ1j, axis=1, keepdims=True) * (1. / X.shape[1])
    W1j = W1j - learning_rate * dW1j
    b1j = b1j - learning_rate * db1j
    W2j = W2j - learning_rate * dW2j
    b2j = b2j - learning_rate * db2j
    W3j = W3j - learning_rate * dW3j
    b3j = b3j - learning_rate * db3j
    W4j = W4j - learning_rate * dW4j
    b4j = b4j - learning_rate * db4j

    cost_j = cost_function(Y, A4j, m, epsilon)
    cost_j = np.squeeze(cost_j)
    cost_list_j.append(cost_j)

    if i % print_num == 0 or i == epochs - 1:
        print("Cost for optimized backprop after iteration {}: {}".format(i, cost_j))

    if cost_j < cost_lim_down:
        last_i_j = i
        print("Cost for optimized backprop after iteration {}: {}".format(i, cost_j))
        break
    else:
        continue

t3 = time.process_time() - t2
print("Time elapsed for optimized backprop: ", t3)
print()

# Plot cost functions
plt.figure(figsize=(10, 10))
plt.suptitle('Conventional Backprop', fontsize=25)
plt.title(f"Time elapsed: {t1: .0f} seconds \n Number of epochs: {last_i_n}", fontsize=15)
plt.plot(list(range(last_i_n + 1)), cost_list_n, color='blue')
plt.ylim(bottom=cost_lim_down, top=cost_lim_up)
plt.xlim(left=0, right=last_i_n + 5)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Cost", fontsize=15)
plt.show()

print()

plt.figure(figsize=(10, 10))
plt.suptitle('Optimized Backprop', fontsize=25)
plt.title(f"Time elapsed: {t3:.0f} seconds \n Number of epochs: {last_i_j}", fontsize=15)
plt.plot(list(range(last_i_j + 1)), cost_list_j, color='red')
plt.ylim(bottom=cost_lim_down, top=cost_lim_up)
plt.xlim(left=0, right=last_i_n + 5)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Cost", fontsize=15)
plt.show()
