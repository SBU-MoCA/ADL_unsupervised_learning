import os.path

from matplotlib import pyplot as plt

from helper import datetime_from_str
import numpy as np
from datetime import datetime
import os

# define time range
time_ranges = [
    (datetime.strptime("2023-06-29 18:25", "%Y-%m-%d %H:%M"), datetime.strptime("2023-06-29 19:02", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-06-29 19:02", "%Y-%m-%d %H:%M"), datetime.strptime("2023-06-29 19:37", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-06-29 20:01", "%Y-%m-%d %H:%M"), datetime.strptime("2023-06-29 20:26", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-06-29 20:49", "%Y-%m-%d %H:%M"), datetime.strptime("2023-06-29 21:14", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-06-29 21:25", "%Y-%m-%d %H:%M"), datetime.strptime("2023-06-29 21:37", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-07-03 19:23", "%Y-%m-%d %H:%M"), datetime.strptime("2023-07-03 19:51", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-07-03 20:45", "%Y-%m-%d %H:%M"), datetime.strptime("2023-07-03 21:04", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-07-03 21:16", "%Y-%m-%d %H:%M"), datetime.strptime("2023-07-03 21:33", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-07-03 20:05", "%Y-%m-%d %H:%M"), datetime.strptime("2023-07-03 20:25", "%Y-%m-%d %H:%M")),
    (datetime.strptime("2023-07-07 19:05", "%Y-%m-%d %H:%M"), datetime.strptime("2023-07-07 19:33", "%Y-%m-%d %H:%M"))
]

nodes = ["b8-27-eb-63-ae-61", "b8-27-eb-4e-d2-eb", "b8-27-eb-b4-f8-c2"]
path = "ADL_data/2024-09-18_timestamps/"

# check if the timestamp is within the time ranges
def is_in_time_ranges(dt, time_ranges):
    """
    check if the timestamp is within the time ranges
    """
    for start, end in time_ranges:
        if start <= dt <= end:
            return True
    return False

def aggregate_sensor_timestamp(path, time_ranges, nodes):
    """
    read timestamp.txt from pi nodes, aggregate timestamps within the time ranges and save them to a new file
    """
    # path = "ADL_data/2024-09-18_timestamps/"
    for node in nodes:
        print(node)
        node_path = path + node
        files  = os.listdir(node_path)
        dts = []
        for file in files:
            filename = os.path.join(node_path, file)
            with open(filename, "r") as file:
                lines = file.readlines()
                for ts in lines:
                    dt = datetime_from_str(ts)
                    if not is_in_time_ranges(dt, time_ranges):
                        continue
                    dts.append(dt)
        if len(dts) == 0:
            continue

        np.savetxt(f"{path}{node}_timestamps.txt", dts, fmt="%s")


def aggregate_DB_timestamp(path, time_ranges, nodes):
    """
    aggregate timestamps from cloud InfluxDB, and save them to a new file
    """
    for node in nodes:
        print(node)
        sessions = os.listdir("ADL_data/")
        dts = []
        for session in sessions:
            if "ADL" in session:
                print(session)
                file = f"ADL_data/{session}/{node}_timestamp.txt"
                with open(file, "r") as f:
                    lines = f.readlines()
                    timestamps = [datetime_from_str(line) for line in lines]
                    dts.extend(timestamps)
        np.savetxt(f"ADL_data/{node}_timestamps_DB.txt", dts, fmt="%s")
        

def compare(list_1, list_2, filename):
    ''' compare the two lists of timestamp

    Arg:
        list_1: timestamps from sensor, one timestamp per mqtt message.
        list_2: timestamps from cloud InfluxDB, one timestamp per frame.
    Return:

    '''
    # sort is necessary.
    list_1.sort()
    list_2.sort()
    if len(list_1) == 0:
        return 0
    i = 0
    j = 0
    loss = []  # lost frames
    y = [0] * len(list_1)  # 0: received, 1: lost
    while True:
        lp = list_1[i].replace(microsecond=0)
        ls = list_2[j].replace(microsecond=0)
        if lp == ls:
            i += 1
            j += 1
        elif lp < ls:  # data loss
            i += 1
            loss.append(lp)
            if i == len(list_1):
                break
            y[i] = 1
            # print(f"loss: {lp}")
        else:
            j += 1
        if i == len(list_1) or j == len(list_2):
            break
    print("finish")
    loss.extend(list_1[i:])
    y[i:] = [1] * (len(list_1) - i)
    print("# of loss: {}, data frame on sensor: {}, loss rate: {}".format(len(loss), len(list_1), len(loss)/len(list_1)))
    
    for start, end in time_ranges:
        # Filter timestamps within the current time range
        range_indices = [i for i, ts in enumerate(list_1) if start <= ts <= end]
        if not range_indices:
            continue

        range_y = [y[i] for i in range_indices]
        range_list_1 = [list_1[i] for i in range_indices]

        if 1 not in range_y:
            print(f"{filename} no data loss from {start} to {end}")
            continue
        # Plot the data for the current time range
        plot_x = []
        plot_y = []
        for idx in range(len(range_y)):
            if range_y[idx] == 1:
                if idx == 0 or range_y[idx - 1] == 0:
                    plot_x.append(range_list_1[idx])
                    plot_y.append(1)
                elif idx == len(range_y) - 1 or range_y[idx + 1] == 0:
                    plot_x.append(range_list_1[idx])
                    plot_y.append(1)

        plt.plot(plot_x, plot_y, marker='o')

        # Annotate the first and last loss points
        for idx in range(len(range_y)):
            if range_y[idx] == 1:
                if idx == 0 or range_y[idx - 1] == 0:
                    plt.annotate(f"{range_list_1[idx].strftime('%H:%M:%S')}", (range_list_1[idx], 1.25), textcoords="offset points", xytext=(0,10), ha='center')
                elif idx == len(range_y) - 1 or range_y[idx + 1] == 0:
                    plt.annotate(f"{range_list_1[idx].strftime('%H:%M:%S')}", (range_list_1[idx], 1), textcoords="offset points", xytext=(0,10), ha='center')

        plt.xlabel("Time")
        plt.ylabel("is loss")
        plt.ylim(0, 2)
        plt.yticks([0, 1], ["received", "lost"])
        plt.xticks([range_list_1[0], range_list_1[-1]], [range_list_1[0].strftime('%Y-%m-%d %H:%M'), range_list_1[-1].strftime('%Y-%m-%d %H:%M')])
        plt.title(f"Loss from {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
        plt.savefig(f"{filename}_loss_{start.strftime('%Y%m%d_%H%M')}_to_{end.strftime('%Y%m%d_%H%M')}.png")
        plt.cla()




nodes = ["b8-27-eb-4e-d2-eb"]
# aggregate_sensor_timestamp("ADL_data/2024-09-18_timestamps/", time_ranges, ["b8-27-eb-b4-f8-c2"])

# start, end = time_ranges[-1][0], time_ranges[-1][1]
for node in nodes:
    print(node)
    file_db = f"ADL_data/{node}_timestamps_DB.txt"
    file_sensor = f"ADL_data/2024-09-18_timestamps/{node}_timestamps.txt"
    timestamps_db = []
    timestamps_sensor = []
    with open(file_db, "r") as f:
        lines = f.readlines()
        # timestamps_db = [datetime_from_str(line) for line in lines if start < datetime_from_str(line) < end]
        timestamps_db = [datetime_from_str(line) for line in lines]
    with open(file_sensor, "r") as f:
        lines = f.readlines()
        # timestamps_sensor = [datetime_from_str(line) for line in lines if start < datetime_from_str(line) < end]
        timestamps_sensor = [datetime_from_str(line) for line in lines]
    print(len(timestamps_db), len(timestamps_sensor))
    compare(timestamps_sensor, timestamps_db, node)