import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F


def load_dataFile(path_list, train_size=0.8):
	
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	X_val = []
	y_val = []
	for path in path_list:
		# path1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
		x1 = np.load(os.path.join(path, "X_4.npy"))[:, np.newaxis, :, :]
		y1 = np.load(os.path.join(path, "Y_4.npy")) - 1
		x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=train_size, random_state=42)
		x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, train_size=train_size, random_state=42)

		if len(X_train) == 0:
			X_train = x1_train
			y_train = y1_train
			X_test = x1_test
			y_test = y1_test
			X_val = x1_val
			y_val = y1_val
		else:
			X_train = np.vstack((x1_train, X_train))
			y_train = np.concatenate((y1_train, y_train))
		
			X_test = np.vstack([x1_test, X_test])
			y_test = np.concatenate([y1_test, y_test])

			X_val = np.vstack([x1_val, X_val])
			y_val = np.concatenate([y1_val, y_val])
	
	X = np.vstack([X_train, X_test, X_val])
	x_mean = np.mean(X)
	x_std = np.std(X)
	X_train = (X_train - x_mean) / x_std
	X_test = (X_test - x_mean) / x_std
	X_val = (X_val - x_mean) / x_std


	np.save("X_test.npy", X_test)
	np.save("y_test.npy", y_test)

	return torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()

def resize_tensor(train_data, test_data, size=(224, 224)):
	resized_train_data = F.interpolate(train_data, size=size,mode='bilinear', align_corners=False)
	resized_test_data = F.interpolate(test_data, size=size,mode='bilinear', align_corners=False)
	return resized_train_data, resized_test_data


def convert_to_dataset(train_data, train_label, test_data, test_label):
	train_dataset = TensorDataset(train_data, train_label)
	test_dataset = TensorDataset(test_data, test_label)
	return train_dataset, test_dataset


def data_loader(train_dataset, test_dataset, batch_size=16):
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
	return train_loader, test_loader


def wrapper_dataLoader(path_list, train_size=0.8, batch_size=16, if_resize=False, if_replicate_channels=False):
	train_data, train_label, test_data, test_label = load_dataFile(path_list, train_size=train_size)
	if if_resize:
		train_data, test_data = resize_tensor(train_data, test_data)
	if if_replicate_channels:
		train_data = train_data.repeat(1, 3, 1, 1)
		test_data = test_data.repeat(1, 3, 1, 1)
	train_set, test_set = convert_to_dataset(train_data, train_label, test_data, test_label)
	train_loader, test_loader = data_loader(train_set, test_set, batch_size=batch_size)
	return train_loader, test_loader


if __name__ == "__main__":
	path_list = ["/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6",
				"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5",
				"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4",
				"/home/mengjingliu/Vid2Doppler/data/2023_11_17/HAR3",
				"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR2"]
	wrapper_dataLoader(path_list, batch_size=16, if_resize=True, if_replicate_channels=True)




