import sys
sys.path.append("/home/mengjingliu/ADL_unsupervised_learning/")

import csv
import os.path
from helper import datetime_from_str
import numpy as np

from config import nodes


def load_csv_new(path, filename):
	"""
	This is for new version of load_csv, where real and imaginary are stored in the same field ("complex_data", which has [real1, imag1, real2, imag2, ...]) in 2024.
	"""
	
	data = {"complex_data": [], "timestamp": []}
	
	# read csv file
	with open(os.path.join(path, filename), 'r', newline='\n') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		headers = next(csvreader, None)
		print(headers)      # "_time" is the time data is written into database. field "timestamp" is when the data is generated.
		
		for row in csvreader:
			if row == []:
				print("empty row")
				continue
			try:
				data["complex_data"].append(row[6])
				data["timestamp"].append(row[5])
			except Exception as e:
				print(row)
				raise(e)
			
	# concatenate
	complex_data = data["complex_data"]
	timestamp = data["timestamp"]
	
	# format conversion
	dt = [datetime_from_str(t) for t in timestamp]
	complex_data = np.array([[float(v) for v in d.split(',')] for d in complex_data])
	complex_float = np.array([[d[i] + d[i+1]*1j for i in range(0, len(d), 2)] for d in complex_data])
	
	# sort
	indices = sorted(
		range(len(dt)),
		key=lambda index: dt[index]
	)
	
	complex_sort = complex_float[indices, :]
	
	# dt.sort()
	# return imag_sort, real_sort, dt
	node_id = filename.split('.')[0]
	np.save(os.path.join(path, "{}_complex.npy".format(node_id)), complex_sort)
	dt.sort()
	dt_str = [str(v) for v in dt]
	with open(os.path.join(path, "{}_timestamp.txt".format(node_id)), 'w') as f:
		f.write('\n'.join(dt_str))





def load_csv(path, filename):
	"""
	This funciton is for old version of load_csv, where we can real and imaginary stored in two seperate fields in 2023.

	load InfluxDB data from csv file and save data in imaginary.npy, real.npy, timestamp.txt. Sort data by timestamps.
	This can save storage space by 1/3 compared to csv format.
	Args:
		path: path of csv file and the path to save data in .npy
		filename: filename of csvfile

	Returns:

	"""
	
	data = {"imaginary": [], "real": [], "timestamp": []}
	data_last_frame = {"imaginary": [], "real": [], "timestamp": []}
	
	# read csv file
	with open(os.path.join(path, filename), 'r', newline='\n') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		headers = next(csvreader, None)
		print(headers)      # "_time" is the time data is written into database. field "timestamp" is when the data is generated.
		
		header_line = False
		for row in csvreader:
			if len(row) == 11:      # with "last frame" tag
				data_last_frame[row[-4]].append(row[-5])
			else:   # wo "last frame" tag
				if len(row) == 0:
					header_line = True
				elif header_line:
					print(row)
					header_line = False
				else:
					try:
						data[row[-3]].append(row[-4])
					except Exception as e:
						print(row)
	
	# concatenate
	imag = data_last_frame["imaginary"]
	imag.extend(data["imaginary"])
	real = data_last_frame["real"]
	real.extend(data["real"])
	timestamp = data_last_frame["timestamp"]
	timestamp.extend(data["timestamp"])
	
	# format conversion
	dt = [datetime_from_str(t) for t in timestamp]
	imag_float = np.array([[float(v) for v in line.split(';')] for line in imag])
	real_float = np.array([[float(v) for v in line.split(';')] for line in real])
	
	# sort
	indices = sorted(
		range(len(dt)),
		key=lambda index: dt[index]
	)
	
	imag_sort = imag_float[indices, :]
	real_sort = real_float[indices, :]
	
	# dt.sort()
	# return imag_sort, real_sort, dt
	node_id = filename.split('.')[0]
	np.save(os.path.join(path, "{}_imaginary.npy".format(node_id)), imag_sort)
	np.save(os.path.join(path, "{}_real.npy".format(node_id)), real_sort)
	dt.sort()
	dt_str = [str(v) for v in dt]
	with open(os.path.join(path, "{}_timestamp.txt".format(node_id)), 'w') as f:
		f.write('\n'.join(dt_str))
	

if __name__ == "__main__":
	# path = "/home/mengjingliu/ADL_unsupervised_learning/ADL_data/RQAKB1_ADL_1"
	# for node in nodes:
	# 	print(node)
	# 	filename = f"{node}.csv"
	# 	try:
	# 		load_csv(path, filename)
	# 	except Exception as e:
	# 		print(e)
	# load_csv_new("/home/mengjingliu/ADL_unsupervised_learning/ADL_data/8F33UK", "b8-27-eb-ec-77-37.csv")
	data = np.load("ADL_data/8F33UK/b8-27-eb-ec-77-37_complex.npy")
	print(data)
	print(data.shape)
