from config import nodes_info
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from helper import first_peak
import matplotlib
from scipy.signal import find_peaks
from helper import load_txt_to_datetime, seg_index, datetime_from_str, color_scale, seg_index_new, load_segment_file_to_datetime_new
from export_data_from_influxDB.load_data_from_csv import load_csv, load_csv_new
from config import nodes_info
import subprocess
from datetime import timedelta, timezone
import math

# load _timestamp.txt file, check the start and end time of the data
def load_timestamp_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_time = datetime_from_str(lines[0].strip())
        end_time = datetime_from_str(lines[-1].strip())
    return start_time, end_time, len(lines)

n = []
path = "ADL_data/SB-46951W-2"
files = os.listdir(path)
files.sort()
for file in files:
    if file.endswith("_timestamp.txt"):
        print(file.split("_")[0])
        start_time, end_time, lines = load_timestamp_file(os.path.join(path, file))
        print(start_time, end_time, lines)
        n.append(lines)
print("average # of frames", np.array(n).mean())
print("# of frames expected", (end_time - start_time).total_seconds() * 120)
print((np.array(n).mean())/((end_time - start_time).total_seconds() * 120))