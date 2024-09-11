import time
import os
import numpy as np
from helper import datetime_from_str
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

path = "ADL_data/YpyRw1_ADL_2"
node_id = "b8-27-eb-02-d4-0b"
ts_uwb_file = os.path.join(path, node_id + '_timestamp.txt')
ts_rgb_file = path + "/rgb_ts.txt"


def check_fr_UWB(ts_uwb_file):
    dts = []
    with open(ts_uwb_file, "r") as file:
        lines = file.readlines()
        for ts in lines:
            dt = datetime_from_str(ts)
            dts.append(dt)

    avg_fr = (len(dts) + 1) / (dts[-1] - dts[0]).seconds
    print(f"average frame rate of UWB {node_id}: ", avg_fr)
    plt.plot(dts, np.arange(1, len(dts) + 1))
    plt.xlabel("time (second)")
    plt.ylabel("# of frames")
    plt.title(f"UWB data {node_id}: {avg_fr:.2f}fps")
    plt.savefig(path + f"/UWB_frame_rate_{node_id}.png")
    plt.cla()

    


def check_fr_rgb(ts_rgb_file):
    ts_rgb = np.loadtxt(ts_rgb_file, dtype=float)

    avg_fr = (len(ts_rgb) + 1) / (ts_rgb[-1] - ts_rgb[0])
    print(f"average frame rate of rgb {node_id}: ", avg_fr)

    plt.plot(ts_rgb, np.arange(1, len(ts_rgb) + 1))
    plt.xlabel("time (second)")
    plt.ylabel("# of frames")
    plt.title(f"Video data {node_id}: {avg_fr:.2f}fps")
    plt.savefig(path + f"/rgb_frame_rate_{node_id}.png")

check_fr_UWB(ts_uwb_file)
check_fr_rgb(ts_rgb_file)