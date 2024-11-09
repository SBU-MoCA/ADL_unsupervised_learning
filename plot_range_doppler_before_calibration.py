"""
plot range doppler in video, plot segment line of each activity.

varaibles:
node_idx: the index of sensor node
path: the path of the UWB data session
path_seg: the path of the segment file
filename_seg_shifted: the name of the segment file for this session after time shift
script_file: the name of the script file
fps_uwb: the frame rate of UWB sensor. By default, it is 120 FPS. But different UWB sensor may have slights different frame rates.
	run check_fr_UWB() in check_framerate.py to check the frame rate of UWB sensor.

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


node_idx = 16
path = "/home/mengjingliu/ADL_Detection/ADL_data/YpyRw1_ADL_2"

path_seg = "/home/mengjingliu/ADL_Detection/ADL_data/2023-07-03-segment"
filename_seg_shifted = "YpyRw1_shifted.txt"

script_file = "ADL_data/2023-07-03-segment/script.txt"
fps_uwb = 116.1

############################################################################################################
node_id = nodes_info[node_idx]["id"]
left, right = nodes_info[node_idx]["range"]
filename = f"{node_id}.csv"
doppler = []
doppler_bin_num = 32
DISCARD_BINS = [15, 16]

# load raw data
complex_file = os.path.join(path, node_id + '_complex.npy')
ts_uwb_file = os.path.join(path, node_id + '_timestamp.txt')

if not os.path.exists(complex_file):
	load_csv(path, filename)
ts_UWB = load_txt_to_datetime(path, ts_uwb_file)	# load timestamps

data_complex = np.load(complex_file)

# load timestamps of each UWB baseband data frame
dt = load_txt_to_datetime(path, ts_uwb_file)    # load timestamp of each data frame

indices, acts = seg_index(dt, path_seg, filename_seg_shifted, 0, None)
print(indices)

acts = []
with open(script_file, "r") as f:
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
        out_vid = cv2.VideoWriter(os.path.join(path, f'range_doppler_sensor_{node_idx}.mpg'),
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

subprocess.call(['ffmpeg', '-y', '-i', os.path.join(path, f'range_doppler_sensor_{node_idx}.mpg'),
				 os.path.join(path, f'range_doppler_sensor_{node_idx}.mp4')])

subprocess.call(['rm', os.path.join(path, f'range_doppler_sensor_{node_idx}.mpg')])
exit(0)
