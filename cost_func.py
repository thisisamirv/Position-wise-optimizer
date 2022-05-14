# Import packages
import numpy as np

# Cost functions (forward)
def BCE_loss(Y_hat, Y):
	m = Y_hat.shape[1]
	cost = (-1 / m) * (np.dot(Y, np.log(Y_hat+1e-5).T) + np.dot(1-Y, np.log(1-Y_hat+1e-5).T))
	cost = np.squeeze(cost)
	return cost

def MSE_loss(Y_hat, Y):
	m = Y_hat.shape[1]
	cost = 1 / m * np.power(Y-Y_hat, 2)
	cost = np.squeeze(cost)
	return cost

# Cost functions (backward)
def BCE_loss_backward(Y_hat, Y):
	dA_prev = - (np.divide(Y, Y_hat+1e-5) - np.divide(1-Y, 1-Y_hat+1e-5))
	return dA_prev

def MSE_loss_backward(Y_hat, Y):
	dA_prev = Y - Y_hat
	return dA_prev

