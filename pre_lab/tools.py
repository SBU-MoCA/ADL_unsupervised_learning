import numpy as np
import os
from sklearn.metrics import accuracy_score


def load_data(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-23_bedroom/46-d7"):
	if os.path.exists(os.path.join(data_path, 'doppler_original_segmented_aligned.npy')):
		print("load aligned data.")
		data = np.load(os.path.join(data_path, 'doppler_original_segmented_aligned.npy'))
		label = np.loadtxt(os.path.join(data_path, 'label_segmented_aligned.txt')).astype(int)
	else:
		print("load non-aligned data.")
		data = np.load(os.path.join(data_path, 'doppler_original_segmented.npy'))
		label = np.loadtxt(os.path.join(data_path, 'label_segmented.txt')).astype(int)
	return data, label


def chose_central_rangebins(d_fft, DISCARD_BINS=[14, 15, 16], threshold=0.1, ARM_LEN=30):
	fft_discard = np.copy(d_fft)
	# discard velocities around 0
	for j in DISCARD_BINS:
		fft_discard[j, :] = np.zeros(188)
	# select range bins with the highest energy
	mmax = np.max(fft_discard)
	row, column = np.where(fft_discard == mmax)
	row, column = row[0], column[0]
	left = column
	right = column + 1
	if column > 0:
		for left in range(column - 1, 0, -1):
			if np.max(fft_discard[:, left]) < threshold * mmax:
				break
	if column < 187:
		for right in range(column + 1, 188, 1):
			if np.max(fft_discard[:, right]) < threshold * mmax:
				break
	left = max(left, column - ARM_LEN)
	right = min(right, column + ARM_LEN)
	return left, right


def select_Minpts(data, k):
	nn_dist = []
	k_dist = []
	for i in range(data.shape[0]):
		dist = np.power(np.power(data[i] - data, 2).sum(axis=1), 0.5)
		dist.sort()
		k_dist.append(dist[k])
		nn_dist.append(dist)
	return np.array(k_dist), np.array(nn_dist)


def align_label(pre, label, cluster_number):
	pre = pre + 1
	pre += cluster_number

	# label[label==4] = 3
	number = np.bincount(label)
	cum_num = np.cumsum(number)

	# pre[pre==(np.argmax(np.bincount(pre[:2580])))] = 3
	# pre[pre>4] = 4
	pre[pre == (np.argmax(np.bincount(pre[:cum_num[1]])))] = 1
	# pre[pre==(np.argmax(np.bincount(pre[cum_num[1]:cum_num[2]])))] = 2
	# pre[pre==(np.argmax(np.bincount(pre[number[1]+number[2]:number[1]+number[2]+number[3]])))] = 3
	pre[pre > 2] = 2
	print(accuracy_score(label, pre))