from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import os
from sklearn.metrics import accuracy_score


def load_data(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/46-d7/"):
	data = np.load(os.path.join(data_path, 'doppler_original_segmented.npy'))
	label = np.loadtxt(os.path.join(data_path, 'label_segmented.txt')).astype(int)
	return data, label




# data, label = load_data()
# data[:, 24:27, :] = np.zeros((len(data), 3, 188))
# data = data[:, :, :42]
# data = np.reshape(data, [len(data), 50*42])
# mm = np.mean(data)
# std = np.std(data)
# data_1d_1 = np.array([d.flatten() for d in data])
# data_1d_1 = (data_1d_1 - mm) / std

data, label = load_data(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/fd-bf")
# data = data[:, :, :42]
data_1d_2 = np.array([d.flatten() for d in data])
# data, label = load_data(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/d0-0b")
# data_1d_3 = np.array([d.flatten() for d in data])
# ll = min(len(data_1d_1), len(data_1d_2))
# data = np.hstack((data_1d_1[:ll, :], data_1d_2[:ll, :]))
model = KMeans(n_clusters=4, random_state=0, n_init=5).fit(data_1d_2)
# model = DBSCAN(eps=1, min_samples=10).fit(data_1d)
pre = model.labels_ + 1
print(max(pre))
print(accuracy_score(label, pre))


