# Import packages
import numpy as np
import copy
from dataset import *
import os

# Network structure
input_l = int(input("Specify the number of hidden layers: "))
layers_dim = []
for i in range (input_l):
	n = int(input(f"Specify the number of neurons for hidden layer {i+1}: "))
	layers_dim.append(n)

layers_dim.insert(0, n_features)
layers_dim.append(1)
L = len(layers_dim)

# Recommended values
if data=="1":
	recom_lr = 0.0005
	recom_cost_lim = 0.69
elif data=="2":
	recom_lr = 0.000001
	recom_cost_lim = 0.0003

# Get hyperparameters
epochs = int(input("Specify the number of epochs: "))
learning_rate = float(input(f"Specify the learning rate (Recommended for this dataset: <{recom_lr}): "))
cost_lim = float(input(f"Specify a limit for the validation cost (Recommended for this dataset: <{recom_cost_lim}): "))
cost_list_GD = []
cost_list_PW = []
cost_list_GD_val = []
cost_list_PW_val = []

# Print data structure, network structure, and hyperparameters
os.system('cls' if os.name == 'nt' else 'clear')
print("====================================")
print("Data and network structure: ")
print(f"Dataset: {data_name}")
print(f"Number of training examples: {m}")
print(f"Number of validation examples: {m_val}")
print(f"Training Y shape: {Y_train.shape}")
print(f"Training X shape: {X_train.shape}")
print(f"Validation Y shape: {Y_val.shape}")
print(f"Validation X shape: {X_val.shape}")
print(f"Number of layers: {L}")
print(f"Number of neurons in each layer: {layers_dim}")
print(f"Learning rate: {learning_rate}")
print(f"Specified number of epochs: {epochs}")
print(f"Goal for the validation cost value: {cost_lim}")
print("====================================")
print()


# Initialize parameters
params = {}
for l in range(1, L):
	params['W'+str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) / np.sqrt(layers_dim[l-1])
	params['b'+str(l)] = np.zeros((layers_dim[l], 1))

params_GD = copy.deepcopy(params)
params_PW = copy.deepcopy(params)

# Functions
def gradient(dZ, A_prev):
	dW = np.dot(dZ, A_prev.T) * (1 / A_prev.shape[1])
	db = np.sum(dZ, axis=1, keepdims=True) * (1 / A_prev.shape[1])
	return dW, db

def update(W, b, dW, db, learning_rate):
	W = W - np.dot(learning_rate, dW)
	b = b - np.dot(learning_rate, db)
	return W, b
