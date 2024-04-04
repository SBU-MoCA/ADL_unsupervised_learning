"""
get the start and stop datetime of each activity.
input the start and stop time of an activity in the video, with [start minutes, start second, stop minute, stop second].
load the timestamp file of video, consisting of the timestamp of each frame.
print the list of start and stop datetimes.

"""

import os.path

from helper import datetime_from_str
import numpy as np
from scipy.signal import find_peaks
import datetime
from datetime import timedelta, datetime


def segment_video(rbs_ts_file, minutes_start, seconds_start, minutes_stop, seconds_stop, fr=24):
	"""
return the start datetime and stop datetime of each activity, in local timezone,
given the start time and stop time of the activity in the video and the frame timestamps (rgb_ts.txt) of the video.
	Args:
		rbs_ts_file: timestamp file of video
		minutes_start:
		seconds_start:
		minutes_stop:
		seconds_stop:
		fr: frame rate of video
	"""
	# dur = timedelta(minutes=minutes, seconds=seconds)
	frame_num_start = (minutes_start * 60 + seconds_start) * fr
	frame_num_stop = (minutes_stop * 60 + seconds_stop) * fr
	with open(rbs_ts_file) as file:
		ts = file.readlines()
		if frame_num_start < 1:
			frame_num_start = 1
		if frame_num_stop > len(ts):
			print("frame_num_stop (= {}) > len(ts) (= {})".format(frame_num_stop, len(ts)))
			frame_num_stop = len(ts)
		dt_start = datetime.fromtimestamp(float(ts[int(frame_num_start) - 1]))
		dt_stop = datetime.fromtimestamp(float(ts[int(frame_num_stop) - 1]))
		print("start:{},stop:{}".format(str(dt_start), str(dt_stop)))
		return dt_start, dt_stop


def convert_manual_segment_file_to_AppFormat(manual_seg_filename, rbs_ts_file, app_seg_path, app_seg_filename, fr=24):
	"""
	load manual segment file, including the start time and end time of each activity in the video
	it is not absolute datetime, but the time progress in video
	manual segment file format: "activity name - start: minute, second,stop: minute, second"
	APP segment file format: "activity name - start: datetime,stop: datetime"
	
	Args:
		manual_seg_filename: manual segmentation file
		rbs_ts_file: video timestamp file
		app_seg_path: path to save App format segment file
		app_seg_filename: App format segment filename
		fr: frame rate of video
	"""
	
	segs = []
	acts = []
	entries = []
	with open(manual_seg_filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			act = line.strip('\n').split(' - ')[0]
			acts.append(act)
			ss = line.strip('\n').split(' - ')[1].split(',')
			start_min = int(ss[0][7:])
			start_sec = int(ss[1][1:])
			stop_min = int(ss[2][6:])
			stop_sec = int(ss[3][1:])
			segs.append([start_min, start_sec, stop_min, stop_sec])
	
	for seg in segs:
		start, stop = segment_video(rbs_ts_file, seg[0], seg[1], seg[2], seg[3], fr=fr)
		entries.append(f"{act} - start: {start},stop: {stop}")
	
	with open(os.path.join(app_seg_path, app_seg_filename), "w") as f:
		for entry in entries:
			f.write(entry)
			
	print("manual segment file converted to App format segmentation file successfully!")
	print(f"manual segment file: {manual_seg_filename}")
	print(f"App format seggment file: {app_seg_path}/{app_seg_filename}")
	
	
# if __name__ == "__main__":
	# input the start and stop time of an activity in the video. format: [start minutes, start second, stop minute, stop second]
	# segs = [
	# 	[5, 39, 5, 47],
	# 	[5, 49, 5, 59],
	# 	[6, 16, 7, 1],
	# 	[7, 2, 7, 9],
	# 	[7, 13, 7, 22],
	# 	[7, 26, 7, 33]
	#
	# ]
	# for seg in segs:
	# 	segment_video("/home/mengjingliu/ADL_unsupervised_learning/ADL_data/YhsHv0_ADL_1/rgb_ts.txt", seg[0], seg[1],
	# 	              seg[2], seg[3])

