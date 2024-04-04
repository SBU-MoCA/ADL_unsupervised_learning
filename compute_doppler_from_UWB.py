"""
load segment file of the data session, load .csv datafile to .npy.
read each segment of baseband data and plot range profile and Doppler.


"""

import os
# import subprocess
# import random
# import math
# import time
import numpy as np
# import shutil
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
import cv2
from helper import first_peak
import matplotlib
from scipy.signal import find_peaks
from helper import load_txt_to_datetime, seg_index, apply_lowpass_filter
from load_data_from_csv import load_csv


# need to decide the range
left = 0
right = 188

path = "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/YhsHv0_ADL_1"
filename = "b8-27-eb-dc-a9-b5.csv"
node_num = 3

path_seg = "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/2023-07-03-segment"
filename_seg = "2023-06-29-17-29-50_YhsHv0_ADL_1.txt"

node_id = filename.split('.')[0]

doppler = []
doppler_bin_num = 32
DISCARD_BINS = [15, 16]
fps_uwb = 180

# load raw data
imag_file = os.path.join(path, node_id + '_imaginary.npy')
real_file = os.path.join(path, node_id + '_real.npy')
ts_uwb_file = os.path.join(path, node_id + '_timestamp.txt')

if not os.path.exists(imag_file):
	load_csv(path, filename)
ts_UWB = load_txt_to_datetime(path, ts_uwb_file)
# ts_uwb = np.loadtxt(ts_uwb_file)  # load timestamps
data_imag = np.load(imag_file)      # load imaginary part
data_real = np.load(real_file)      # load real part
number_of_frames = min(len(data_imag), len(data_real))
print("imaginary file: {} lines, real file: {} lines".format(len(data_imag), len(data_real)))
data_imag = data_imag[0:number_of_frames]
data_real = data_real[0:number_of_frames]

data_complex = data_real + 1j * data_imag  # compute complex number

# locate moving object
range_profile = np.abs(data_complex)
hp = sns.heatmap(range_profile)
plt.show()

indices, acts = seg_index(path, filename, path_seg, filename_seg, 1, 6)
# activity = 1
for ss, act in zip(indices, acts):
	start, stop = ss[0], ss[1]
	range_profile_seg = range_profile[start:stop, :]
	# locate moving object
	std_dis = np.std(range_profile_seg, axis=0)
	std_dis = apply_lowpass_filter(std_dis, 0.5)
	# left, right = first_peak(std_dis)
	# print("left: {}, right: {}.".format(left, right))
	# plt.plot(np.arange(0, std_dis.shape[0], 1), std_dis)
	# plt.axvline(x=left, color='r')
	# plt.axvline(x=right, color='r')
	# plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
	# plt.ylabel("standard deviation")
	# plt.xlabel("range bin")
	# plt.title("activity {}".format(act))
	# plt.savefig(os.path.join(path, "first_peak_{}_{}.png".format(act, node_num)))
	# plt.show()

	# plot range profile
	# mean_dis = np.mean(range_profile_seg, axis=0)
	# range_profile_seg = range_profile_seg - mean_dis
	hp = sns.heatmap(np.transpose(range_profile_seg))
	# plt.axvline(x=left, color='r')
	# plt.axvline(x=right, color='r')
	plt.title("range profile of activity {}".format(act))
	plt.xlabel("time (1/{} second)".format(fps_uwb))
	plt.ylabel("range bin")
	hp.figure.savefig(os.path.join(path, "range_profile_{}_{}.png".format(act, node_num)))
	plt.show()
	
	# compute doppler
	doppler_from_UWB = []
	doppler_1d = []
	for i in range(start + doppler_bin_num, stop, 2):
		d_fft = np.abs(np.fft.fft(data_complex[i - doppler_bin_num:i, :], axis=0))  # FFT
		d_fft = np.fft.fftshift(d_fft, axes=0)  # shifts
		# d_fft[DISCARD_BINS, :] = np.zeros((len(DISCARD_BINS), d_fft.shape[1]))
		doppler_from_UWB.append(d_fft)

		fft_gt = np.copy(d_fft[:, left:right])
		fft_gt[DISCARD_BINS, :] = np.zeros((len(DISCARD_BINS), right - left))
		# sum over range
		fft_gt = np.sum(fft_gt, axis=1)
		doppler_1d.append(fft_gt)
	# doppler_1d = (doppler_1d - np.min(doppler_1d)) / (np.max(doppler_1d) - np.min(doppler_1d))
	hp = sns.heatmap(np.transpose(doppler_1d))
	plt.title("doppler of activity {}".format(act))
	plt.ylabel("doppler")
	plt.xlabel("time")
	hp.figure.savefig(os.path.join(path, "doppler_{}_{}.png".format(act, node_num)))
	plt.show()
	
	activity += 1
