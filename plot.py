# Import packages
import matplotlib.pyplot as plt

# Plot cost functions
def plot_cost(t1, last_epoch_GD_train, last_epoch_GD, cost_list_GD, cost_list_GD_val, t2, last_epoch_PW_train, last_epoch_PW, cost_list_PW, cost_list_PW_val, cost_lim):
	plt.figure(figsize=(10, 10))
	plt.subplot(2, 2, 1)
	plt.title(f"Gradient descent optimizer \n Time elapsed: {t1: .1f} seconds \n Number of epochs: {last_epoch_GD}", fontsize=15)
	plt.plot(list(range(last_epoch_GD_train+1)), cost_list_GD, color='blue')
	plt.ylim(bottom=cost_lim)
	plt.xlim(left=0, right=last_epoch_GD_train+5)
	plt.xlabel("Epochs", fontsize=15)
	plt.ylabel("Cost (Training)", fontsize=15)
	plt.subplot(2, 2, 2)
	plt.title(f"Position-wise optimizer \n Time elapsed: {t2: .1f} seconds \n Number of epochs: {last_epoch_PW}", fontsize=15)
	plt.plot(list(range(last_epoch_PW_train+1)), cost_list_PW, color='red')
	plt.ylim(bottom=cost_lim)
	plt.xlim(left=0, right=last_epoch_GD_train+5)
	plt.xlabel("Epochs", fontsize=15)
	plt.ylabel("Cost (Training)", fontsize=15)
	plt.subplot(2, 2, 3)
	plt.plot(list(range(last_epoch_GD+1)), cost_list_GD_val, color='blue')
	plt.ylim(bottom=cost_lim)
	plt.xlim(left=0, right=last_epoch_GD+5)
	plt.xlabel("Epochs", fontsize=15)
	plt.ylabel("Cost (Validation)", fontsize=15)
	plt.subplot(2, 2, 4)
	plt.plot(list(range(last_epoch_PW+1)), cost_list_PW_val, color='red')
	plt.ylim(bottom=cost_lim)
	plt.xlim(left=0, right=last_epoch_GD+5)
	plt.xlabel("Epochs", fontsize=15)
	plt.ylabel("Cost (Validation)", fontsize=15)
	plt.show()
