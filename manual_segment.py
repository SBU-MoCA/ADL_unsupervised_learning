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
from datetime import timedelta, datetime, timezone
from helper import load_segment_file_to_datetime


def time_shift(app_seg_path, app_seg_filename, shift_minute, shift_second):	
	"""
	Load app segment file, transform from local timezone to UTC timezone.
	Then shift the app time by minute and second to align with video time, 
	"""
	dts = []
	with open(os.path.join(app_seg_path, app_seg_filename), 'r') as f:
		lines = f.readlines()
			
		for line in lines:
			try:
				ss = line.strip('\n').split(',')
			except Exception as e:
				print(line)
			dts.append([datetime_from_str(ss[0][6:]).astimezone(timezone.utc), datetime_from_str(ss[1][5:]).astimezone(timezone.utc)])   # convert timezone to UTC time. default is the local time of the OS

	shifted_segment_file = app_seg_filename.split('.')[0] + "_shifted.txt"
	with open(os.path.join(app_seg_path, shifted_segment_file), 'w') as f:
		for i in range(len(dts)):
			dts[i][0] = dts[i][0] + timedelta(minutes=shift_minute, seconds=shift_second)
			dts[i][1] = dts[i][1] + timedelta(minutes=shift_minute, seconds=shift_second)
			print(f"{i+1}, {dts[i][0].strftime('%Y-%m-%d %H:%M:%S.%f')}, {dts[i][1].strftime('%Y-%m-%d %H:%M:%S.%f')}")
			f.write(f"{i+1}, {dts[i][0].strftime('%Y-%m-%d %H:%M:%S.%f')}, {dts[i][1].strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
	return dts


def segment_video(rgb_ts_file, minutes_start, seconds_start, minutes_stop, seconds_stop, fr=24):
	"""
return the start datetime and stop datetime of each activity, in local timezone,
given the start time and stop time of the activity in the video and the frame timestamps (rgb_ts.txt) of the video.
	Args:
		rgb_ts_file: timestamp file of video
		minutes_start:
		seconds_start:
		minutes_stop:
		seconds_stop:
		fr: frame rate of video. 
		This is different from real time frame rate in real world timeline. the frame rate of video is defined by the video writing code (which is 24 in our case), and the time of the video is different from the real time.
	"""
	# dur = timedelta(minutes=minutes, seconds=seconds)
	frame_num_start = (minutes_start * 60 + seconds_start) * fr
	frame_num_stop = (minutes_stop * 60 + seconds_stop) * fr
	with open(rgb_ts_file) as file:
		ts = file.readlines()
		if frame_num_start < 1:
			frame_num_start = 1
		if frame_num_stop > len(ts):
			print("frame_num_stop (= {}) > len(ts) (= {})".format(frame_num_stop, len(ts)))
			frame_num_stop = len(ts)
		dt_start = datetime.fromtimestamp(float(ts[int(frame_num_start) - 1])) + timedelta(minutes=4, seconds=53)
		dt_stop = datetime.fromtimestamp(float(ts[int(frame_num_stop) - 1])) + timedelta(minutes=4, seconds=53)
		print("start:{},stop:{}".format(str(dt_start), str(dt_stop)))
		return dt_start, dt_stop


def convert_manual_segment_file_to_AppFormat(manual_seg_filename, rgb_ts_file, app_seg_path, app_seg_filename, fr=24):
	"""
	load manual segment file, including the start time and end time of each activity in the video
	it is not absolute datetime, but the time progress in video
	manual segment file format: "activity sequence number - start: minute, second, stop: minute, second"
	APP segment file format: "activity sequence number - start: datetime,stop: datetime"
	
	Args:
		manual_seg_filename: manual segmentation file
		rgb_ts_file: video timestamp file
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
			stop_min = int(ss[2][7:])
			stop_sec = int(ss[3][1:])
			segs.append([start_min, start_sec, stop_min, stop_sec])
	
	for seg in segs:
		start, stop = segment_video(rgb_ts_file, seg[0], seg[1], seg[2], seg[3], fr=fr)
		entries.append(f"{act} - start: {start},stop: {stop}")
	
	with open(os.path.join(app_seg_path, app_seg_filename), "w") as f:
		for entry in entries:
			f.write(entry)
			
	print("manual segment file converted to App format segmentation file successfully!")
	print(f"manual segment file: {manual_seg_filename}")
	print(f"App format seggment file: {app_seg_path}/{app_seg_filename}")
	
	
if __name__ == "__main__":
	# input the start and stop time of an activity in the video. format: [start minutes, start second, stop minute, stop second]
	# rgb_ts_file = "ADL_data/YpyRw1_ADL_2/rgb_ts_sensor_3_camera_1.txt"
	# segs = [
	# 	[0, 45, 0, 49],
	# 	[0, 51, 1, 5],
	# 	[1, 10, 1, 20],
	# 	[1, 22, 1, 27],
	# 	[1, 32, 1, 37],
	# 	[1, 39, 1, 53]
	# ]
	# for seg in segs:
	# 	segment_video(rgb_ts_file, seg[0], seg[1], seg[2], seg[3])

	# segment_video(rgb_ts_file, 17, 0, 1, 1, fr=24)
	time_shift("ADL_data/2023-07-03-segment", "2023-07-03-16-10-44_YpyRw1_ADL_2.txt", -4, -42)
