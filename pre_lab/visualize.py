import numpy as np
import os
import cv2
import matplotlib
from matplotlib import cm
import datetime
import imutils
# from prepare_data import load_segments, find_segment_indices

def load_seg_file(file, flag=False):
	"""
	load segment files from my android app
	Args:
		file:
		flag: whether it's mixed. like take on/off, sit down/stand up

	Returns: 2-d list of paris [datetime_begin, datetime_end]

	"""
	if not os.path.exists(file):
		print("{} doesn't exist.".format(file))
		return []
	with open(file) as f:
		lines = f.readlines()

	if len(lines) == 1:  # start:xxx,start:xxx,
		dt_str = lines[0].split(',')[:-1]
		date_times = [datetime.datetime.strptime(dt[6:], '%Y-%m-%d %H:%M:%S.%f') for dt in dt_str]
		if not flag:
			seg = [[date_times[i], date_times[i + 1]] for i in range(len(date_times) - 1)]
			return seg, []
		else:
			seg_1 = [[date_times[i], date_times[i + 1]] for i in range(0, len(date_times) - 1, 2)]
			seg_2 = [[date_times[i], date_times[i + 1]] for i in range(1, len(date_times) - 1, 2)]
			return seg_1, seg_2
	else:  # start:xxx, stop:xxx\n
		if flag:
			seg_1 = [[datetime.datetime.strptime(line.strip('\n').split(',')[0][6:], '%Y-%m-%d %H:%M:%S.%f'),
			        datetime.datetime.strptime(line.strip('\n').split(',')[1][5:], '%Y-%m-%d %H:%M:%S.%f')] for line in lines]
			seg_2 = [[datetime.datetime.strptime(lines[i].strip('\n').split(',')[1][5:], '%Y-%m-%d %H:%M:%S.%f'),
			          datetime.datetime.strptime(lines[i+1].strip('\n').split(',')[0][6:], '%Y-%m-%d %H:%M:%S.%f')] for i in range(len(lines)-1)]
			return seg_1, seg_2
		else:
			seg = [[datetime.datetime.strptime(line.split(',')[0][6:], '%Y-%m-%d %H:%M:%S.%f'),
			          datetime.datetime.strptime(line.split(',')[1][5:], '%Y-%m-%d %H:%M:%S.%f')] for line in lines]
			return seg, []



def load_segments(seg_path='/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/segmentation'):
	"""
	load multiple segment files from my android app and organize them by labels
	Args:
		data_path:
		seg_path:

	Returns: segments of type [[datetime_start, datetime_stop]] for each class

	"""
	seg1 = []
	seg2 = []
	seg3 = []
	seg4 = []

	# load segment files
	# zh 1
	seg_file = os.path.join(seg_path, '2023-02-09-15-51-33.txt')
	seg1, _ = load_seg_file(seg_file, False)

	# zh 2
	seg_file = os.path.join(seg_path, '2023-02-09-16-00-14.txt')
	seg2, _ = load_seg_file(seg_file, False)

	# zh 3, 4
	seg_file = os.path.join(seg_path, '2023-02-09-16-26-20.txt')
	seg3, seg4 = load_seg_file(seg_file, True)

	seg_file = os.path.join(seg_path, '2023-02-09-16-42-48.txt')
	seg33, seg44 = load_seg_file(seg_file, True)
	seg3.extend(seg33)
	seg4.extend(seg44)

	# lmj 1
	seg_file = os.path.join(seg_path, '2023-02-09-17-56-33.txt')
	seg, _ = load_seg_file(seg_file, False)
	seg1.extend(seg)

	# lmj 2
	seg_file = os.path.join(seg_path, '2023-02-09-18-03-16.txt')
	seg, _ = load_seg_file(seg_file, False)
	seg2.extend(seg)

	# lmj 3, 4
	seg_file = os.path.join(seg_path, '2023-02-09-16-26-20.txt')
	seg33, seg44 = load_seg_file(seg_file, True)
	seg3.extend(seg33)
	seg4.extend(seg44)

	return seg1, seg2, seg3, seg4


def find_segment_indices(seg_dt_start, seg_dt_end, times_dt):
	"""
	traverse datetime of each data frame to find the segment fitting in seg_dt_start and seg_dt_end. return the indices.s

	Args:
		seg_dt_start: datetime
		seg_dt_end:
		times_dt:
		augment:

	Returns: indices of frames which is collected between datetime seg_dt_start, seg_dt_end

	"""
	i = 0
	data_indices = []
	while i < len(times_dt) and times_dt[i] < seg_dt_start:
		i += 1
	while i < len(times_dt) and times_dt[i] <= seg_dt_end:
		data_indices.append(i)
		i += 1
	return data_indices


def color_scale(img, norm,text=None):
	if len(img.shape) == 2:
		img = cm.magma(norm(img),bytes=True)
	# img = imutils.resize(img, height=300)
	if img.shape[2] == 4:
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
	# if text is not None:
	# 	img = cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
	return img


def visualize(data_path, data, label):

	for idx in range(len(data)):
		original_dop = color_scale(data[idx], matplotlib.colors.Normalize(vmin=0, vmax=np.max(data)), label)
		if idx == 0:
			print(original_dop.shape)
			height, width = original_dop.shape[0], original_dop.shape[1]
			print(os.path.join(data_path, 'range_doppler_output_signal_{}.mp4'.format(label)))
			out_vid = cv2.VideoWriter(os.path.join(data_path, 'range_doppler_output_signal_{}.mp4'.format(label)),
			                          cv2.VideoWriter_fourcc(*'mp4v'),
		                          20, (width, height))
		out_vid.write(original_dop)

	out_vid.release()


def visualize_timeline(data_path, seg_path, label):
	data_complex = np.load(os.path.join(data_path, 'data_complex.npy'))
	times = np.loadtxt(os.path.join(data_path, 'times.txt'))
	import datetime
	times_dt = [datetime.datetime.fromtimestamp(time) for time in times]

	seg_all = [[], [], [], []]
	seg_all[0], seg_all[1], seg_all[2], seg_all[3] = load_segments(seg_path)

	k = 0
	for segg in seg_all[2][10:16]:
		k += 1
		seg_start, seg_end = segg[0], segg[1]
		indices = find_segment_indices(seg_start, seg_end, times_dt)
		dc_ = data_complex[indices,:]
		for i in range(0, len(dc_)-50, 50):
			dc = dc_[i:i+50, :]
			d_fft_ = np.abs(np.fft.fft(dc, axis=0))  # FFT
			d_fft = np.fft.fftshift(d_fft_, axes=0)  # shift
			d_fft[24:27] = np.zeros((3, 188))
			d_fft_color = color_scale(d_fft, matplotlib.colors.Normalize(vmin=0, vmax=np.max(d_fft)), label)
			if i == 0:
				print(d_fft_color.shape)
				height, width = d_fft_color.shape[0], d_fft_color.shape[1]
				print(os.path.join(data_path, 'range_doppler_timeline_signal_{}.mp4'.format(label)))
				out_vid = cv2.VideoWriter(os.path.join(data_path, 'range_doppler_timeline_signal_3_{}.mp4'.format(k)),
				                          cv2.VideoWriter_fourcc(*'mp4v'),
			                          10, (width, height))
			out_vid.write(d_fft_color)

		out_vid.release()

	k = 0
	for segg in seg_all[3][10:16]:
		k+= 1
		seg_start, seg_end = segg[0], segg[1]
		indices = find_segment_indices(seg_start, seg_end, times_dt)
		dc_ = data_complex[indices,:]
		for i in range(0, len(dc_)-50, 50):
			dc = dc_[i:i+50, :]
			d_fft_ = np.abs(np.fft.fft(dc, axis=0))  # FFT
			d_fft = np.fft.fftshift(d_fft_, axes=0)  # shift
			d_fft[24:27] = np.zeros((3, 188))
			d_fft_color = color_scale(d_fft, matplotlib.colors.Normalize(vmin=0, vmax=np.max(d_fft)), label)
			if i == 0:
				print(d_fft_color.shape)
				height, width = d_fft_color.shape[0], d_fft_color.shape[1]
				print(os.path.join(data_path, 'range_doppler_timeline_signal_{}.mp4'.format(label)))
				out_vid = cv2.VideoWriter(os.path.join(data_path, 'range_doppler_timeline_signal_4_{}.mp4'.format(k)),
				                          cv2.VideoWriter_fourcc(*'mp4v'),
			                          10, (width, height))
			out_vid.write(d_fft_color)

		out_vid.release()



if __name__ == "__main__":
	# if no file saved, change python environment to vid2dop
	data_path = "/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-23_bedroom/fd-bf/"
	data = np.load(os.path.join(data_path, 'doppler_original_segmented.npy'))
	data = data/0.65
	data[:, 24:27, :] = np.zeros((len(data), 3, 188))
	# data[:, 24:27, :] = np.zeros((len(data), 3, 188))
	seg_path = '/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-23_bedroom/segmentation'
	label = np.loadtxt(os.path.join(data_path, 'label_segmented.txt')).astype(int)
	label[label==4]=3
	number = np.bincount(label)
	cum_num = np.cumsum(number)
	# visualize(data_path, data[:cum_num[1]], "microwave")
	visualize(data_path, data[cum_num[0]:cum_num[1]], "fold")

	visualize(data_path, data[cum_num[1]:cum_num[2]], "closet")

# visualize_timeline(data_path, seg_path, 1234)

	# visualize_timeline(data_path, data[:3031, :, :], 1)
	# visualize_timeline(data_path, data[3031:6331, :, :], 2)
	# visualize_timeline(data_path, data[6331:8911, :, :], 3)
	# visualize_timeline(data_path, data[8911:, :, :], 4)
