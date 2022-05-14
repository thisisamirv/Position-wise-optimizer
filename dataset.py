# Import packages
import numpy as np
import sys

# Import datasets
data = input("Datasets are: 1. Cat vs. non-cat classification, 2. Breast Ultrasound Images dataset, 3. MNIST dataset. Enter desired dataset number: ")

grads_GD = {}
grads_PW = {}
grads_GD_val = {}
grads_PW_val = {}
if data == "1":
	data_name = "Cat vs. non-cat classification"
	import h5py
	train_dataset = h5py.File('CatClassificationTrain.h5', "r")
	X_train = np.array(train_dataset["train_set_x"][:])
	Y_train = np.array(train_dataset["train_set_y"][:])
	Y_train = Y_train.reshape((1, Y_train.shape[0]))
	X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
	X_train = X_train_flatten / 255.
	val_dataset = h5py.File('CatClassificationVal.h5', "r")
	X_val = np.array(val_dataset["test_set_x"][:])
	Y_val = np.array(val_dataset["test_set_y"][:])
	Y_val = Y_val.reshape((1, Y_val.shape[0]))
	X_val_flatten = X_val.reshape(X_val.shape[0], -1).T
	X_val = X_val_flatten / 255.
	task = "classification"
	print_num = 100
elif data == "2":
	data_name = "Breast Ultrasound Images dataset"
	from PIL import Image
	import os
	from sklearn.model_selection import train_test_split
	images_normal = []
	directory1 = 'Dataset_BUSI_with_GT/normal'
	for file_name in os.listdir(directory1):
		if file_name != '.DS_Store':
			path = os.path.join(directory1, file_name)
			img = Image.open(path)
			img = img.resize((300, 300), Image.Resampling.NEAREST)
			arr = np.asarray(img)
			flat_arr = np.ravel(arr)
			images_normal.append(flat_arr)
	X1 = np.stack(images_normal, axis=1)
	Y1 = np.full((1,133), 1)
	images_malignant = []
	directory2 = 'Dataset_BUSI_with_GT/malignant'
	for file_name in os.listdir(directory2):
		if file_name != '.DS_Store':
			path = os.path.join(directory2, file_name)
			img = Image.open(path)
			img = img.resize((300, 300), Image.Resampling.NEAREST)
			arr = np.asarray(img)
			flat_arr = np.ravel(arr)
			images_malignant.append(flat_arr)
	X2 = np.stack(images_malignant, axis=1)
	Y2 = np.full((1,210), 1)
	X = np.concatenate((X1, X2), axis=1)
	Y = np.concatenate((Y1, Y2), axis=1)
	X_train, X_val, Y_train, Y_val = train_test_split(X.T, Y.T, test_size=0.2, shuffle=True)
	X_train = X_train.T
	X_val = X_val.T
	Y_train = Y_train.T
	Y_val = Y_val.T
	task = "classification"
	print_num = 20
else:
	print("Invalid input")
	sys.exit()

grads_GD['A0'] = X_train
grads_PW['A0'] = X_train
grads_GD_val['A0'] = X_val
grads_PW_val['A0'] = X_val

# Initialize some values
m = X_train.shape[1]
n_features = X_train.shape[0]
m_val = X_val.shape[1]

