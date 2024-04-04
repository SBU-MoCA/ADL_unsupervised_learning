import os
# import subprocess
# import random
# import math
# import time
import numpy as np
# import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from math import log2
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
import cv2
from helper import first_peak
import matplotlib
from scipy.signal import find_peaks


def compute_noise(dop, threshold=0.8):
	ind = np.argmax(dop)
	mmax = dop[ind]
	background = []
	if ind > 0:
		for i in range(ind-1, 0, -1):
			if dop[i] < mmax * threshold:
				break
		background.extend(dop[0:i+1].tolist())
	if ind < len(dop):
		for i in range(ind+1, len(dop)):
			if dop[i] < mmax * threshold:
				break
		background.extend(dop[i+1:].tolist())
	return np.mean(np.array(background))


path = "/home/mengjingliu/Vid2Doppler/data/2023_05_04/2023_05_04_18_07_20_mengjing_push"

doppler_1d = np.load(os.path.join(path, "doppler_1d_from_UWB.npy"))
mmax = np.max(doppler_1d)
mmin = np.min(doppler_1d)
doppler_1d = (doppler_1d - mmin) / (mmax - mmin)

sns.heatmap(doppler_1d[:300, :])
plt.show()


eta_ = []
cnt = 0
for dop in doppler_1d:
	# plt.plot(dop)
	# plt.grid()
	# plt.show()
	e_peak = np.max(dop)
	# e_noise = np.mean(dop[dop < e_peak*0.8])
	e_noise = compute_noise(dop)
	eta = log2((e_peak + e_noise) / e_noise)
	eta_.append(eta)
	cnt += 1


