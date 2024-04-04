from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import os
from sklearn.metrics import accuracy_score
from tools import load_data






data, label = load_data(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-23_kitchen/46-d7")
# data[:, 24:27, :] = np.zeros((len(data), 3, 188))
# data = data[:, :, :60]
# data = np.reshape(data, [len(data), 50*42])
# mm = np.mean(data)
# std = np.std(data)
data_1d_1 = np.array([d.flatten() for d in data])
# data_1d_1 = (data_1d_1 - mm) / std

# data, label = load_data(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09_restroom/fd-bf")
# data[:, 24:27, :] = np.zeros((len(data), 3, 188))
# data = data[:, :, :60]
# data_1d_2 = np.array([d.flatten() for d in data])
# data, label = load_data(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09_restroom/d0-0b")
# data[:, 24:27, :] = np.zeros((len(data), 3, 188))
# data = data[:, :, :60]
# data_1d_3 = np.array([d.flatten() for d in data])
# data = np.hstack((data_1d_3, data_1d_1))
# data = np.hstack((np.hstack((data_1d_1, data_1d_2)), data_1d_3))
X = data_1d_1[:5139, :]
model = KMeans(n_clusters=3, random_state=0, n_init=5).fit(X)
centers = model.cluster_centers_
print(centers)
new_X = data_1d_1[4228:5140, :]

# try to detect novel class based on distance from cluster centers and radius.
# Not work. clusters are not of circle shape.
cluster_distance = model.transform(X)
cluster_dis = np.min(cluster_distance, axis=1)
r1 = np.percentile(cluster_dis[:2301], 99)
r2 = np.percentile(cluster_dis[2301:4228], 99)
model_param = model.get_params()

for x in new_X:
	p = model.predict([x])[0]
	distances = model.transform([x])[0]
	# dis = np.min(cluster_dis)
	# print(dis)
	if distances[0] > r1 and distances[1] > r2:
		centers_ex = np.vstack((centers, [x]))
		model_ex = KMeans(n_clusters=3, random_state=0, n_init=5, init=centers_ex).fit(X)


# model = DBSCAN(eps=1, min_samples=10).fit(data_1d)
pre = model.labels_ + 1
pre += 2

# label[label==4] = 3
number = np.bincount(label)
cum_num = np.cumsum(number)

# pre[pre==(np.argmax(np.bincount(pre[:2580])))] = 3
# pre[pre>4] = 4
pre[pre==(np.argmax(np.bincount(pre[:cum_num[1]])))] = 1
# pre[pre==(np.argmax(np.bincount(pre[cum_num[1]:cum_num[2]])))] = 2
# pre[pre==(np.argmax(np.bincount(pre[number[1]+number[2]:number[1]+number[2]+number[3]])))] = 3
pre[pre>2] = 2
print(accuracy_score(label, pre))


