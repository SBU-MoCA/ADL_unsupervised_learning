"""
load segment file of the data session, load .csv datafile to .npy.
plot range doppler in video, plot segment line of each activity.

"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from helper import first_peak
import matplotlib
from scipy.signal import find_peaks
from helper import load_txt_to_datetime, seg_index, apply_lowpass_filter, color_scale, seg_index_new
from export_data_from_influxDB.load_data_from_csv import load_csv, load_csv_new
from config import nodes_info
import subprocess
from datetime import timedelta


def plot_range_doppler(session_path, shifted_seg_file, node_num, shift_sensor_time, doppler_bin_num=32, step_size=5, new_version=False):
	"""
	plot range doppler video of the specific sensor in specific data session.
	:param session_path: the path of the data session, e.g. "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/YpyRw1_ADL_2"
	:param node_num: the index of the sensor in the config file
	:param shift_sensor_time: the time shift of the sensor to align with the video, e.g. [0, -5], add 0 minute and -5 seconds on sensor time
	:param shifted_seg_file: the path of the segment file, which has been shifted to align with video time, e.g. "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/2023-07-03-segment/YpyRw1_shifted.txt"

	"""
	node_id = nodes_info[node_num]["id"]
	left, right = nodes_info[node_num]["range"]
	filename = f"{node_id}.csv"

	doppler = []
	DISCARD_BINS = [15, 16]

	# load raw data
	if not new_version:	# old data in 2023
		imag_file = os.path.join(session_path, node_id + '_imaginary.npy')
		real_file = os.path.join(session_path, node_id + '_real.npy')
		ts_uwb_file = os.path.join(session_path, node_id + '_timestamp.txt')

		if not os.path.exists(imag_file):
			load_csv(session_path, filename)
		data_imag = np.load(imag_file)      # load imaginary part
		data_real = np.load(real_file)      # load real part
		number_of_frames = min(len(data_imag), len(data_real))
		print("imaginary file: {} lines, real file: {} lines".format(len(data_imag), len(data_real)))
		data_imag = data_imag[0:number_of_frames]
		data_real = data_real[0:number_of_frames]

		data_complex = data_real + 1j * data_imag  # compute complex number

		# load timestamps of each UWB baseband data frame
		dt = load_txt_to_datetime(session_path, ts_uwb_file)    # load timestamp of each data frame
		dt = [t + timedelta(minutes=shift_sensor_time[0], seconds=shift_sensor_time[1]) for t in dt] # shift the timestamp to align with the video

		acts = []
		with open("ADL_data/2023-07-03-segment/script.txt", "r") as f:
			lines = f.readlines()
			for line in lines:
				acts.append(line.strip('\n').split('. ')[0])
		indices, acts = seg_index(dt, shifted_seg_file, 0, None)
		print("indices: ", indices)
		print("acts: ", acts)
	else:
		complex_file = os.path.join(session_path, node_id + '_complex.npy')
		ts_uwb_file = os.path.join(session_path, node_id + '_timestamp.txt')

		if not os.path.exists(complex_file):
			load_csv_new(session_path, filename)
		data_complex = np.load(complex_file)
		dt = load_txt_to_datetime(session_path, ts_uwb_file)    # load timestamp of each data frame

		acts = []
		indices, acts = seg_index_new(dt, shifted_seg_file, 0, None)
		print("indices: ", indices)
		print("acts: ", acts)

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
			out_vid = cv2.VideoWriter(os.path.join(session_path, f'range_doppler_sensor_{node_num}.mpg'),
									cv2.VideoWriter_fourcc(*'mp4v'),
									15, (dd.shape[1], dd.shape[0]))
			flag = False
		
		
		# plot the label
		if plot_label:
			while k < len(acts):
				if j < indices[k][0]:
					text = ""
					break
				elif j >= indices[k][0] and j < indices[k][1]:	# indices: the index of each activity in baseband data
					text = acts[k]
					break
				elif j >= indices[k][1]:
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

	subprocess.call(['ffmpeg', '-y', '-i', os.path.join(session_path, f'range_doppler_sensor_{node_num}.mpg'),
					os.path.join(session_path, f'range_doppler_sensor_{node_num}.mp4')])

	subprocess.call(['rm', os.path.join(session_path, f'range_doppler_sensor_{node_num}.mpg')])
	return 0


if __name__ == "__main__":
	# for node_num in range(1, 16):
	# 	if node_num in [10, 11, 12, 16]:	
	# 		continue
	# 	try:
	# 		print(f"sensor: {node_num}")
	# 		plot_range_doppler("/home/mengjingliu/ADL_unsupervised_learning/ADL_data/YpyRw1_ADL_2", "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/2023-07-03-segment/YpyRw1_shifted.txt", node_num, [0, -5])
	# 	except Exception as e:
	# 		print(e)
	# 		continue
	# plot_range_doppler("/home/mengjingliu/ADL_unsupervised_learning/ADL_data/YpyRw1_ADL_2", "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/2023-07-03-segment/YpyRw1_shifted.txt", 2, [0, -5])
	plot_range_doppler("/home/mengjingliu/ADL_unsupervised_learning/ADL_data/8F33UK", "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/2023-07-03-segment/2024-10-15-17-22-48_8F33UK.txt", 
					5, [0, 0], new_version=True)
