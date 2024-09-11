import os.path

import numpy as np
# from scipy.signal import find_peaks
import datetime
from datetime import timedelta, timezone
# from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass_filter(data, normal_cutoff, order=5):
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # b, a = butter_lowpass(cutoff_freq, fs, order=order)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def first_peak(std_dis):
    height = max(np.percentile(std_dis, 50), np.max(std_dis) / 3)
    peaks, _ = find_peaks(std_dis, height=height)
    peaks_, _ = find_peaks(-std_dis)
    
    first_peak = peaks[0]
    second_peak = peaks[1]
    left = 0
    right = second_peak
    for pp in peaks_:
        if pp < first_peak:
            left = max(left, pp)
        else:
            break
    
    print("first peak: {}, left: {}, right: {}".format(first_peak, left, right))
    
    return left, right


def datetime_from_str(str):
    try:
        r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%d %H:%M:%S.%f')
    except Exception as e:
        try:
            r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            try:
                r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%S.%fZ')
            except Exception as e:
                try:
                    r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%SZ')
                except Exception as e:
                    try:
                        r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%S.%fZZ')
                    except Exception as e:
                        print("exception: ", str)
                        return ""
    return r


def load_txt_to_datetime(path, filename):
    # load txt file which has a str of datetime in each line. convert str to datetime, return the list of all lines.
    with open(os.path.join(path, filename), 'r') as f:
        lines = f.readlines()
        return [datetime_from_str(line) for line in lines]


def load_segment_file_to_datetime(path, filename, if_local=True):
    """
    load App format segment file from android app, return datetime list [start, stop].
    APP segment file format: "activity name - start: datetime,stop: datetime"
    
    Args:
        path:
        filename:
        if_local: if the time from android app is local time. if it is, set timezone to UTC.

    Returns: datetime list [start, stop], UTC time; activity list

    """
    dt = []
    acts = []
    with open(os.path.join(path, filename), 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            act = line.strip('\n').split(' - ')[0]
            acts.append(act)
            ss = line.strip('\n').split(' - ')[1].split(',')
            dt.append(datetime_from_str(ss[0][7:]).astimezone(timezone.utc), datetime_from_str(ss[1][6:]).astimezone(timezone.utc))
            # if if_local:
            #     dt = [[datetime_from_str(line.split(',')[0][6:]).astimezone(timezone.utc),
            #            datetime_from_str(line.strip('\n').split(',')[1][5:]).astimezone(timezone.utc)]
            #           for line in lines]
            # else:
            #     dt = [[datetime_from_str(line.split(',')[0][6:]), datetime_from_str(line.strip('\n').split(',')[1][5:])]
            #           for line in lines]
    return dt, acts


def dt_delta(dt1, dt2):
    return (dt1 - dt2).seconds * 1e6 + (dt1 - dt2).microseconds


def seg_index(path, filename, path_seg, filename_seg, start_seg=0, stop_seg=None):
    """
load segment file (format: start: datetime, stop: datetime) and UWB frame timestamp file, find the frame indices of each segment in UWB data,
return the frame indices of each segment in UWB date.
    Args:
        path: UWB timestamp file path
        filename: UWB timestamp file name
        path_seg: segment file path
        filename_seg: segment file name
        start_seg: which segment to start
        stop_seg: which segment to stop

    Returns: list of [start, stop], int. the start and stop indices of each segment in UWB timestamp file.
    Assume UWB timestamp, imaginary, real data files are aligned row by row, the returned indices are used to read UWB data for each activity.

    """
    # path = "/home/mengjingliu/ADL_data/I2HSeJ_ADL_1"
    # filename = "b8-27-eb-85-a7-83.csv"
    node_id = filename.split('.')[0]
    ts_uwb_file = os.path.join(path, node_id + '_timestamp.txt')
    
    # path_seg = "/home/mengjingliu/ADL_data/2023-07-03-segment"
    # filename_seg = "2023-07-03-16-50-45_I2HSeJ_ADL_1.txt"
    
    dt = load_txt_to_datetime(path, ts_uwb_file)    # load timestamp of each data frame
    seg, acts = load_segment_file_to_datetime(path_seg, filename_seg)     # load segmentation from android app
    
    i = start_seg
    start, stop = seg[i][0], seg[i][1]
    start_ind = 0
    stop_ind = len(dt)
    indices = []
    next_act = False
    for j in range(len(dt)):    # find the start and stop index for each segment
        t = dt[j]
        if t - dt[j-1] > timedelta(seconds=60):
            print("data loss {} minutes".format((t - dt[j-1]).seconds/60))
        if start_ind == 0:
            if (t < start and dt_delta(start, t) < 1e5) or (start <= t < stop and dt_delta(t, start) <= 1e5):
                start_ind = j
                continue
        if start_ind != 0:
            if (start <= t < stop and dt_delta(stop, t) <= 1e5) or (t >= stop and dt_delta(t, stop) <= 1e5):
                stop_ind = j
                print("activity {}, start: {}, stop: {}, index: {} ~ {}".format(i, str(start), str(stop), start_ind,
                                                                                stop_ind))
                indices.append([start_ind, stop_ind])
                next_act = True
            if t >= stop and (t - stop) >= timedelta(seconds=3):
                print("activity {}, data loss {} seconds around stop point".format(i, (t - dt[j-1]).seconds))
                next_act = True
            if next_act:
                start_ind = 0
                stop_ind = len(dt)
                i += 1
                if (stop_seg is not None and i > stop_seg) or (stop_seg is None and i >= len(seg)):
                    break
                else:
                    start, stop = seg[i][0], seg[i][1]
                next_act = False
            
    return indices, acts


def find_index(start, stop, dt):
    """
    Args:
        start: start datetime of an activity
        stop: stop datetime of an activity
        dt: datetime lists of UWB data

    Returns: index

    """
    start_ind = None
    stop_ind = None
    for j in range(len(dt)):  # find the start and stop index for each segment
        t = dt[j]
        if t < start:
            continue
        if start <= t < stop and start_ind is None:
            start_ind = j
            continue
        if t >= stop and j == 0:
            print("error.")
            return None, None
        if t >= stop and stop_ind is None:
            stop_ind = j
            return start_ind, stop_ind
    return start_ind, j

    