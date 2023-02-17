import numpy as np
import os
import cv2
import matplotlib
from matplotlib import cm
import imutils


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


if __name__ == "__main__":
	data_path = "/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/46-d7/"
	data = np.load(os.path.join(data_path, 'doppler_original_segmented.npy'))[:, :, :60]
	data[:, 24:27, :] = np.zeros((len(data), 3, 60))

	visualize(data_path, data[:2020, :, :], 1)
	visualize(data_path, data[2021:4221, :, :], 2)
	visualize(data_path, data[4221:5941, :, :], 3)
	visualize(data_path, data[5941:, :, :], 4)
