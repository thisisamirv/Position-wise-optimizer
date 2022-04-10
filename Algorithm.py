#!/usr/local/bin/python3

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Choose dataset
data = input("Enter dataset number (1. Cat vs. non-cat classification, 2. CIFAR-10 dataset, 3. MNIST dataset): ")
if data == "1":
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
    cost_lim = 0.1
    learning_rate = 0.0075
    print_num = 100
elif data == "2":
    data_name = "CIFAR-10 dataset"
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
    X, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]
    n_rows = len(train_labels)
    n_cols = train_labels.max() + 1
    Y = np.zeros((n_rows, n_cols), dtype='uint8')
    Y[np.arange(n_rows), train_labels] = 1
    X = X.T
    Y = Y.T
    N = X.shape[1]
    out_dim = 10
    cost_lim = 3.230
    learning_rate = 0.0075
    print_num = 20
elif data == "3":
    data_name = "MNIST dataset"
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
    Y = Y.reshape(-1)
    Y = (np.eye(10)[Y]).T
    X = X.reshape(X.shape[0], -1).T
    Y = Y.astype(np.float32)
    X = X.astype(np.float32)
    N = X.shape[1]
    out_dim = 10
    cost_lim = 3.2494
    learning_rate = 0.000075
    print_num = 50
else:
    print("Invalid input")
    sys.exit()

# Model parameters
layer_dims = [X.shape[0], 20, 7, 5, out_dim]
epochs = 5000
epsilon = 1e-5

def cost_function(Y, A4, N, epsilon):
    cost = (-1 / N) * np.sum(np.multiply(Y, np.log(A4 + epsilon)) + np.multiply(1 - Y, np.log(1 - A4 + epsilon)))
    return cost

# Classification layer
if data == "1":
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
np.random.seed(1)

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

W1_PW = np.copy(W1)
W2_PW = np.copy(W2)
W3_PW = np.copy(W3)
W4_PW = np.copy(W4)
b1_PW = np.copy(b1)
b2_PW = np.copy(b2)
b3_PW = np.copy(b3)
b4_PW = np.copy(b4)

cost_list_GD = []
cost_list_PW = []

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
        print(f"Cost for gradient descent optimizer after epoch {i}: {cost_GD: .4f}")
    elif cost_GD < cost_lim or i == epochs - 1:
        last_i_GD = i
        print(f"Cost for gradient descent optimizer after epoch {i}: {cost_GD: .4f}")
        break
    else:
        continue

t1 = time.process_time() - t0
print(f"Time elapsed for gradient descent optimizer: {t1: .1f} seconds")
print("====================================")
print()

# Position-wise optimizer
t2 = time.process_time()

for i in range(epochs):
    Z1_PW = np.dot(W1_PW, X) + b1_PW
    A1_PW = np.maximum(0, Z1_PW)
    Z2_PW = np.dot(W2_PW, A1_PW) + b2_PW
    A2_PW = np.maximum(0, Z2_PW)
    Z3_PW = np.dot(W3_PW, A2_PW) + b3_PW
    A3_PW = np.maximum(0, Z3_PW)
    Z4_PW = np.dot(W4_PW, A3_PW) + b4_PW
    A4_PW = class_layer(Z4_PW)

    dZ4_PW = A4_PW - Y
    dW4_PW = np.dot(dZ4_PW, A3_PW.T) * (1. / A3_PW.shape[1])
    db4_PW = np.sum(dZ4_PW, axis=1, keepdims=True) * (1. / A3_PW.shape[1])
    W4_PW = W4_PW - learning_rate * dW4_PW
    b4_PW = b4_PW - learning_rate * db4_PW
    Z4_PW = np.dot(W4_PW, A3_PW) + b4_PW
    A4_PW = class_layer(Z4_PW)

    dZ4_PW = A4_PW - Y
    dW4_PW = np.dot(dZ4_PW, A3_PW.T) * (1. / A3_PW.shape[1])
    db4_PW = np.sum(dZ4_PW, axis=1, keepdims=True) * (1. / A3_PW.shape[1])
    dA3_PW = np.dot(W4_PW.T, dZ4_PW)
    dZ3_PW = np.array(dA3_PW, copy=True)
    dZ3_PW[Z3_PW <= 0] = 0
    dW3_PW = np.dot(dZ3_PW, A2_PW.T) * (1. / A2_PW.shape[1])
    db3_PW = np.sum(dZ3_PW, axis=1, keepdims=True) * (1. / A2_PW.shape[1])
    W3_PW = W3_PW - learning_rate * dW3_PW
    b3_PW = b3_PW - learning_rate * db3_PW
    W4_PW = W4_PW - learning_rate * dW4_PW
    b4_PW = b4_PW - learning_rate * db4_PW
    Z3_PW = np.dot(W3_PW, A2_PW) + b3_PW
    A3_PW = np.maximum(0, Z3_PW)
    Z4_PW = np.dot(W4_PW, A3_PW) + b4_PW
    A4_PW = class_layer(Z4_PW)

    dZ4_PW = A4_PW - Y
    dW4_PW = np.dot(dZ4_PW, A3_PW.T) * (1. / A3_PW.shape[1])
    db4_PW = np.sum(dZ4_PW, axis=1, keepdims=True) * (1. / A3_PW.shape[1])
    dA3_PW = np.dot(W4_PW.T, dZ4_PW)
    dZ3_PW = np.array(dA3_PW, copy=True)
    dZ3_PW[Z3_PW <= 0] = 0
    dW3_PW = np.dot(dZ3_PW, A2_PW.T) * (1. / A2_PW.shape[1])
    db3_PW = np.sum(dZ3_PW, axis=1, keepdims=True) * (1. / A2_PW.shape[1])
    dA2_PW = np.dot(W3_PW.T, dZ3_PW)
    dZ2_PW = np.array(dA2_PW, copy=True)
    dZ2_PW[Z2_PW <= 0] = 0
    dW2_PW = np.dot(dZ2_PW, A1_PW.T) * (1. / A1_PW.shape[1])
    db2_PW = np.sum(dZ2_PW, axis=1, keepdims=True) * (1. / A1_PW.shape[1])
    W2_PW = W2_PW - learning_rate * dW2_PW
    b2_PW = b2_PW - learning_rate * db2_PW
    W3_PW = W3_PW - learning_rate * dW3_PW
    b3_PW = b3_PW - learning_rate * db3_PW
    W4_PW = W4_PW - learning_rate * dW4_PW
    b4_PW = b4_PW - learning_rate * db4_PW
    Z2_PW = np.dot(W2_PW, A1_PW) + b2_PW
    A2_PW = np.maximum(0, Z2_PW)
    Z3_PW = np.dot(W3_PW, A2_PW) + b3_PW
    A3_PW = np.maximum(0, Z3_PW)
    Z4_PW = np.dot(W4_PW, A3_PW) + b4_PW
    A4_PW = class_layer(Z4_PW)

    dZ4_PW = A4_PW - Y
    dW4_PW = np.dot(dZ4_PW, A3_PW.T) * (1. / A3_PW.shape[1])
    db4_PW = np.sum(dZ4_PW, axis=1, keepdims=True) * (1. / A3_PW.shape[1])
    dA3_PW = np.dot(W4_PW.T, dZ4_PW)
    dZ3_PW = np.array(dA3_PW, copy=True)
    dZ3_PW[Z3_PW <= 0] = 0
    dW3_PW = np.dot(dZ3_PW, A2_PW.T) * (1. / A2_PW.shape[1])
    db3_PW = np.sum(dZ3_PW, axis=1, keepdims=True) * (1. / A2_PW.shape[1])
    dA2_PW = np.dot(W3_PW.T, dZ3_PW)
    dZ2_PW = np.array(dA2_PW, copy=True)
    dZ2_PW[Z2_PW <= 0] = 0
    dW2_PW = np.dot(dZ2_PW, A1_PW.T) * (1. / A1_PW.shape[1])
    db2_PW = np.sum(dZ2_PW, axis=1, keepdims=True) * (1. / A1_PW.shape[1])
    dA1_PW = np.dot(W2_PW.T, dZ2_PW)
    dZ1_PW = np.array(dA1_PW, copy=True)
    dZ1_PW[Z1_PW <= 0] = 0
    dW1_PW = np.dot(dZ1_PW, X.T) * (1. / X.shape[1])
    db1_PW = np.sum(dZ1_PW, axis=1, keepdims=True) * (1. / X.shape[1])
    W1_PW = W1_PW - learning_rate * dW1_PW
    b1_PW = b1_PW - learning_rate * db1_PW
    W2_PW = W2_PW - learning_rate * dW2_PW
    b2_PW = b2_PW - learning_rate * db2_PW
    W3_PW = W3_PW - learning_rate * dW3_PW
    b3_PW = b3_PW - learning_rate * db3_PW
    W4_PW = W4_PW - learning_rate * dW4_PW
    b4_PW = b4_PW - learning_rate * db4_PW

    cost_PW = cost_function(Y, A4_PW, N, epsilon)
    cost_PW = np.squeeze(cost_PW)
    cost_list_PW.append(cost_PW)

    if i % print_num == 0:
        print(f"Cost for position-wise optimizer after epoch {i}: {cost_PW: .4f}")
    elif cost_PW < cost_lim or i == epochs - 1:
        last_i_PW = i
        print(f"Cost for position-wise optimizer after epoch {i}: {cost_PW: .4f}")
        break
    else:
        continue

t3 = time.process_time() - t2
print(f"Time elapsed for position-wise optimizer: {t3: .1f} seconds")
print("====================================")
print()

# Plot cost functions
plt.figure(figsize=(10, 10))
plt.suptitle('Gradient Descent Optimizer', fontsize=25)
plt.title(f"Time elapsed: {t1: .0f} seconds \n Number of epochs: {last_i_GD}", fontsize=15)
plt.plot(list(range(last_i_GD + 1)), cost_list_GD, color='blue')
plt.ylim(bottom=cost_lim)
plt.xlim(left=0, right=last_i_GD + 5)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Cost", fontsize=15)
plt.show()

print()

plt.figure(figsize=(10, 10))
plt.suptitle('Position-Wise Optimizer', fontsize=25)
plt.title(f"Time elapsed: {t3: .0f} seconds \n Number of epochs: {last_i_PW}", fontsize=15)
plt.plot(list(range(last_i_PW + 1)), cost_list_PW, color='red')
plt.ylim(bottom=cost_lim)
plt.xlim(left=0, right=last_i_GD + 5)
plt.xlabel("Epochs", fontsize=15)
plt.ylabel("Cost", fontsize=15)
plt.show()
