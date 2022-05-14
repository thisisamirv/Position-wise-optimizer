# Import packages
import numpy as np

# Activation functions (forward)
def linear(W, A_prev, b):
	Z = np.dot(W, A_prev) + b
	return Z

def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	return A

def relu(Z):
	A = np.maximum(0,Z)
	return A

def softmax(Z):
	Z_max = np.max(Z, axis=1, keepdims=True)
	Z_exp = np.exp(Z - Z_max)
	Z_sum = np.sum(Z_exp, axis=1, keepdims=True)
	A = np.divide(Z_exp, Z_sum)
	return A

# Activation functions (backward)
def linear_backward(W, dZ):
	dA_prev = np.dot(W.T, dZ)
	return dA_prev

def sigmoid_backward(dA, Z):
	sig = sigmoid(Z)
	dZ = dA * sig * (1 - sig)
	return dZ

def relu_backward(dA, Z):
	dZ = np.array(dA, copy=True)
	dZ[Z<=0] = 0
	return dZ

def softmax_backward(dA, Z):
	sft = softmax(Z)
	#sft_sum = np.sum(dA * sft, axis=1, keepdims=True)
	#dZ = sft * (dA - sft_sum)
	dZ = sft * (dA - (dA * sft).sum(axis=1)[:,None])
	return dZ
