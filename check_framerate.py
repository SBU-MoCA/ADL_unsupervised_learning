import time
import os
import numpy as np
from helper import datetime_from_str
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


def check_fr_UWB(node_id, start_time=None, end_time=None):
    """
    node_id: node index
    start_time: start time of the time period to check the frame rate. str, e.g. "2023-07-03 12:00:00"
    stop_time: end time of the time period to check the frame rate. str, e.g. "2023-07-03 12:05:00"
    """
    ts_uwb_file = os.path.join(path, node_id + '_timestamp.txt')    # timestamp of UWB data, downloaded from cloud DB. 
    dts = []
    with open(ts_uwb_file, "r") as file:
        lines = file.readlines()
        for ts in lines:
            dt = datetime_from_str(ts)
            if dt < datetime_from_str(start_time):
                continue
            if dt > datetime_from_str(end_time):
                break
            dts.append(dt)

    avg_fr = len(dts) / (dts[-1] - dts[0]).seconds
    print(f"average frame rate of UWB {node_id}: ", avg_fr)
    plt.plot(dts, np.arange(1, len(dts) + 1))
    plt.xlabel("time (second)")
    print(dts[-1] - dts[0])
    plt.xticks([dts[0], dts[-1]], [str(dts[0]).split('.')[0], str(dts[-1]).split('.')[0]])
    plt.ylabel("# of frames")
    plt.title(f"UWB data {node_id}: {avg_fr:.2f}fps")
    plt.savefig(path + f"/UWB_frame_rate_{node_id}.png")
    plt.cla()

    


def check_fr_rgb(node_id):
    ts_rgb_file = path + "/rgb_ts.txt"
    ts_rgb = np.loadtxt(ts_rgb_file, dtype=float)

    avg_fr = (len(ts_rgb) + 1) / (ts_rgb[-1] - ts_rgb[0])
    print(f"average frame rate of rgb {node_id}: ", avg_fr)

    plt.plot(ts_rgb, np.arange(1, len(ts_rgb) + 1))
    plt.xlabel("time (second)")
    plt.xticks([ts_rgb[0], ts_rgb[-1]], [str(ts_rgb[0]), str(ts_rgb[-1])])
    plt.ylabel("# of frames")
    plt.title(f"Video data {node_id}: {avg_fr:.2f}fps")
    plt.savefig(path + f"/rgb_frame_rate_{node_id}.png")
    plt.cla()


from config import nodes
# nodes = ["b8-27-eb-63-ae-61"]
path = "ADL_data/RQAKB1_ADL_1"
for node in nodes:
    print(node)
    try:
        check_fr_UWB(node)
    except Exception as e:
        print(e)

