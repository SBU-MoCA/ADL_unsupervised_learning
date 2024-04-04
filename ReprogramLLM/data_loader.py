import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch


def data_loader(path_list, batch_size=16, train_size=0.8):
	
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	for path in path_list:
		# path1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
		x1 = np.load(os.path.join(path, "X_4.npy"))[:, np.newaxis, :, :]
		y1 = np.load(os.path.join(path, "Y_4.npy")) - 1
		x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=train_size, random_state=42)
		
		# path2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
		# x2 = np.load(os.path.join(path2, "X_4.npy"))
		# y2 = np.load(os.path.join(path2, "Y_4.npy"))
		# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, random_state=42)
		#
		# path3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
		# x3 = np.load(os.path.join(path3, "X_4.npy"))
		# y3 = np.load(os.path.join(path3, "Y_4.npy"))
		# _, x3_test, _, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=42)
		#
		if len(X_train) == 0:
			X_train = x1_train
			y_train = y1_train
			X_test = x1_test
			y_test = y1_test
		else:
			X_train = np.vstack((x1_train, X_train))
			y_train = np.concatenate((y1_train, y_train))
		
			X_test = np.vstack([x1_test, X_test])
			y_test = np.concatenate([y1_test, y_test])
		
	train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	
	test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
	
	return train_loader, test_loader
