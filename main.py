#!/usr/local/bin/python3

# Import packages and scripts
import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
import time
from dataset import *
from act_func import *
from cost_func import *
from nn import *
from plot import *

# Gradient descent optimizer
t1 = time.process_time()
for i in range(epochs+1):
	## Forward pass
	for l in range(1, L):
		if l==L-1:
			grads_GD['Z'+str(l)] = linear(params_GD['W'+str(l)], grads_GD['A'+str(l-1)], params_GD['b'+str(l)])
			grads_GD['A'+str(l)] = sigmoid(grads_GD['Z'+str(l)])
		else:
			grads_GD['Z'+str(l)] = linear(params_GD['W'+str(l)], grads_GD['A'+str(l-1)], params_GD['b'+str(l)])
			grads_GD['A'+str(l)] = relu(grads_GD['Z'+str(l)])
	
	## Backward pass
	grads_GD['dA'+str(L-1)] = BCE_loss_backward(grads_GD['A'+str(L-1)], Y_train)
	grads_GD['dZ'+str(L-1)] = sigmoid_backward(grads_GD['dA'+str(L-1)], grads_GD['Z'+str(L-1)])
	grads_GD['dW'+str(L-1)], grads_GD['db'+str(L-1)] = gradient(grads_GD['dZ'+str(L-1)], grads_GD['A'+str(L-2)])
	for l in reversed(range(1, L-1)):
		grads_GD['dA'+str(l)] = linear_backward(params_GD['W'+str(l+1)], grads_GD['dZ'+str(l+1)])
		grads_GD['dZ'+str(l)] = relu_backward(grads_GD['dA'+str(l)], grads_GD['Z'+str(l)])
		grads_GD['dW'+str(l)], grads_GD['db'+str(l)] = gradient(grads_GD['dZ'+str(l)], grads_GD['A'+str(l-1)])
	
	## Update parameters
	for l in range(1, L):
		params_GD['W'+str(l)], params_GD['b'+str(l)] = update(params_GD['W'+str(l)], params_GD['b'+str(l)], grads_GD['dW'+str(l)], grads_GD['db'+str(l)], learning_rate)
	
	## Validation
	for l in range(1, L):
		if l==L-1:
			grads_GD_val['Z'+str(l)] = linear(params_GD['W'+str(l)], grads_GD_val['A'+str(l-1)], params_GD['b'+str(l)])
			grads_GD_val['A'+str(l)] = sigmoid(grads_GD_val['Z'+str(l)])
		else:
			grads_GD_val['Z'+str(l)] = linear(params_GD['W'+str(l)], grads_GD_val['A'+str(l-1)], params_GD['b'+str(l)])
			grads_GD_val['A'+str(l)] = relu(grads_GD_val['Z'+str(l)])
	
	## Compute cost
	cost_GD = BCE_loss(grads_GD['A'+str(L-1)], Y_train)
	cost_GD_val = BCE_loss(grads_GD_val['A'+str(L-1)], Y_val)
	cost_list_GD_val.append(cost_GD_val)
	
	if cost_GD > cost_lim or i == epochs:
		cost_list_GD.append(cost_GD)
		last_epoch_GD_train = i
	
	## Print cost
	if i % print_num == 0:
		print(f"Training cost for gradient descent optimizer after epoch {i}: {cost_GD: .4f}")
		print(f"Validation cost for gradient descent optimizer after epoch {i}: {cost_GD_val: .4f}")
		print()
	elif cost_GD_val < cost_lim or i == epochs:
		last_epoch_GD = i
		print(f"Training cost for gradient descent optimizer after epoch {i}: {cost_GD: .4f}")
		print(f"Validation cost for gradient descent optimizer after epoch {i}: {cost_GD_val: .4f}")
		print()
		break
	
t1 = time.process_time() - t1
print(f"Time elapsed for gradient descent optimizer: {t1: .1f} seconds")
print("====================================")
print()

# Position-wise optimizer
t2 = time.process_time()
for l in range(1, L):
	## First full forward path
	if l==L-1:
		grads_PW['Z'+str(l)] = linear(params_PW['W'+str(l)], grads_PW['A'+str(l-1)], params_PW['b'+str(l)])
		grads_PW['A'+str(l)] = sigmoid(grads_PW['Z'+str(l)])
	else:
		grads_PW['Z'+str(l)] = linear(params_PW['W'+str(l)], grads_PW['A'+str(l-1)], params_PW['b'+str(l)])
		grads_PW['A'+str(l)] = relu(grads_PW['Z'+str(l)])
for i in range(epochs+1):
	## Position-wise optimization
	for j in range(1, L):
		grads_PW['dA'+str(L-1)] = BCE_loss_backward(grads_PW['A'+str(L-1)], Y_train)
		grads_PW['dZ'+str(L-1)] = sigmoid_backward(grads_PW['dA'+str(L-1)], grads_PW['Z'+str(L-1)])
		grads_PW['dW'+str(L-1)], grads_PW['db'+str(L-1)] = gradient(grads_PW['dZ'+str(L-1)], grads_PW['A'+str(L-2)])
		for l in reversed(range(L-j, L)):
			if l != L-1:
				grads_PW['dA'+str(l)] = linear_backward(params_PW['W'+str(l+1)], grads_PW['dZ'+str(l+1)])
				grads_PW['dZ'+str(l)] = relu_backward(grads_PW['dA'+str(l)], grads_PW['Z'+str(l)])
				grads_PW['dW'+str(l)], grads_PW['db'+str(l)] = gradient(grads_PW['dZ'+str(l)], grads_PW['A'+str(l-1)])
			else:
				pass
		for l in range(L-j, L):
			params_PW['W'+str(l)], params_PW['b'+str(l)] = update(params_PW['W'+str(l)], params_PW['b'+str(l)], grads_PW['dW'+str(l)], grads_PW['db'+str(l)], learning_rate)
		for l in range(L-j, L):
			if l==L-1:
				grads_PW['Z'+str(l)] = linear(params_PW['W'+str(l)], grads_PW['A'+str(l-1)], params_PW['b'+str(l)])
				grads_PW['A'+str(l)] = sigmoid(grads_PW['Z'+str(l)])
			else:
				grads_PW['Z'+str(l)] = linear(params_PW['W'+str(l)], grads_PW['A'+str(l-1)], params_PW['b'+str(l)])
				grads_PW['A'+str(l)] = relu(grads_PW['Z'+str(l)])
	
	## Validation
	for l in range(1, L):
		if l==L-1:
			grads_PW_val['Z'+str(l)] = linear(params_PW['W'+str(l)], grads_PW_val['A'+str(l-1)], params_PW['b'+str(l)])
			grads_PW_val['A'+str(l)] = sigmoid(grads_PW_val['Z'+str(l)])
		else:
			grads_PW_val['Z'+str(l)] = linear(params_PW['W'+str(l)], grads_PW_val['A'+str(l-1)], params_PW['b'+str(l)])
			grads_PW_val['A'+str(l)] = relu(grads_PW_val['Z'+str(l)])
	
	## Compute cost
	cost_PW = BCE_loss(grads_PW['A'+str(L-1)], Y_train)
	cost_PW_val = BCE_loss(grads_PW_val['A'+str(L-1)], Y_val)
	cost_list_PW_val.append(cost_PW_val)
	
	if cost_PW > cost_lim or i == epochs:
		cost_list_PW.append(cost_PW)
		last_epoch_PW_train = i
	
	## Print cost
	if i % print_num == 0:
		print(f"Training cost for position-wise optimizer after epoch {i}: {cost_PW: .4f}")
		print(f"Validation cost for position-wise optimizer after epoch {i}: {cost_PW_val: .4f}")
		print()
	elif cost_PW_val < cost_lim or i == epochs:
		last_epoch_PW = i
		print(f"Training cost for position-wise optimizer after epoch {i}: {cost_PW: .4f}")
		print(f"Validation cost for position-wise optimizer after epoch {i}: {cost_PW_val: .4f}")
		print()
		break

t2 = time.process_time() - t2
print(f"Time elapsed for position-wise optimizer: {t2: .1f} seconds")
print("====================================")
print()

# Plot costs
plot_cost(t1, last_epoch_GD_train, last_epoch_GD, cost_list_GD, cost_list_GD_val, t2, last_epoch_PW_train, last_epoch_PW, cost_list_PW, cost_list_PW_val, cost_lim)

