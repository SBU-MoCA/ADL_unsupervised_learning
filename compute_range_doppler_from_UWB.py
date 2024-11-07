"""
load segment file of the data session, load .csv datafile to .npy.
plot range doppler in video, plot segment line of each activity.

"""

import os
import subprocess
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
from helper import load_txt_to_datetime, seg_index, apply_lowpass_filter, color_scale
from export_data_from_influxDB.load_data_from_csv import load_csv
from datetime import timedelta
from config import nodes_info


node_num = 10
node_id = nodes_info[node_num]["id"]
left, right = nodes_info[node_num]["range"]
path = "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/YpyRw1_ADL_2"
filename = f"{node_id}.csv"


path_seg = "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/2023-07-03-segment"
filename_seg = "YpyRw1_shifted.txt"

shift_sensor_time = [0, -5] # add 0 minute and -5 seconds to align with the video

node_id = filename.split('.')[0]

doppler = []
doppler_bin_num = 32
DISCARD_BINS = [15, 16]
fps_uwb = 116.1

# load raw data
imag_file = os.path.join(path, node_id + '_imaginary.npy')
real_file = os.path.join(path, node_id + '_real.npy')
ts_uwb_file = os.path.join(path, node_id + '_timestamp.txt')

if not os.path.exists(imag_file):
	load_csv(path, filename)
ts_UWB = load_txt_to_datetime(path, ts_uwb_file)	# load timestamps
data_imag = np.load(imag_file)      # load imaginary part
data_real = np.load(real_file)      # load real part
number_of_frames = min(len(data_imag), len(data_real))
print("imaginary file: {} lines, real file: {} lines".format(len(data_imag), len(data_real)))
data_imag = data_imag[0:number_of_frames]
data_real = data_real[0:number_of_frames]

data_complex = data_real + 1j * data_imag  # compute complex number

# load timestamps of each UWB baseband data frame
dt = load_txt_to_datetime(path, ts_uwb_file)    # load timestamp of each data frame
dt = [t + timedelta(minutes=shift_sensor_time[0], seconds=shift_sensor_time[1]) for t in dt] # shift the timestamp to align with the video

indices, acts = seg_index(dt, path_seg, filename_seg, 0, None)
print(indices)

acts = []
with open("ADL_data/2023-07-03-segment/script.txt", "r") as f:
	lines = f.readlines()
	for line in lines:
		acts.append(line.strip('\n').split('. ')[0])
print(acts)

step_size = 5		# 120 / 24 = 5
doppler = []
for i in range(indices[0][0], indices[-1][-1], step_size):		
	d_fft = np.abs(np.fft.fft(data_complex[i - doppler_bin_num:i, left:right], axis=0))  # FFT
	d_fft = np.fft.fftshift(d_fft, axes=0)  # shifts

	d_fft[14:18, :] = np.zeros((4, d_fft.shape[1]))
	doppler.append(d_fft)

doppler = np.array(doppler)
# mmax = np.percentile(doppler, 99.99)
mmax = np.max(doppler)
mmin = np.min(doppler)

plot_label = True
flag = True
k = 0		# plot the k-th activity
j = indices[0][0] - doppler_bin_num	# index of the current range-doppler frame in baseband data. (step_size is 5)
for d in doppler:

    dd = color_scale(d[:, left:right], matplotlib.colors.Normalize(vmin=mmin, vmax=mmax), " ")
    if flag:
        out_vid = cv2.VideoWriter(os.path.join(path, f'range_doppler_sensor_{node_num}.mpg'),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                15, (dd.shape[1], dd.shape[0]))
        flag = False
    
	
	# plot the label
    if plot_label:
        if j < indices[k][0]:
            text = ""
        elif j >= indices[k][0] and j < indices[k][1]:	# indices: the index of each activity in baseband data
            text = acts[k]
        elif j >= indices[k][1]:
            text = ""
            k += 1
		
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 50)  # Starting position of the text
        font_scale = 0.5
        color = (255, 255, 255)  # White color in BGR
        thickness = 1  # Thickness of the text
        # Use cv2.putText to add text on the frame
        cv2.putText(dd, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        j += step_size  

    out_vid.write(dd)
		
out_vid.release()

print("frame number:", len(doppler))

print("Video of range doppler generated.")

subprocess.call(['ffmpeg', '-y', '-i', os.path.join(path, f'range_doppler_sensor_{node_num}.mpg'),
				 os.path.join(path, f'range_doppler_sensor_{node_num}.mp4')])

subprocess.call(['rm', os.path.join(path, f'range_doppler_sensor_{node_num}.mpg')])
exit(0)


# activity = 1
for ss, act in zip(indices, acts):
	start, stop = ss[0], ss[1]
	range_profile_seg = range_profile[start:stop, :]
	# locate moving object
	std_dis = np.std(range_profile_seg, axis=0)
	std_dis = apply_lowpass_filter(std_dis, 0.5)
	
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
