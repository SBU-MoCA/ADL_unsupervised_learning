## Segment and load data from InfluxDB
1. watch video on Box and record segmentation information in file.
2. Download **rgb_ts.txt** from Box. 
3. run **manual_segment.py** to get start and stop datatime of each activity. Save segment result in nodeID-DataSessionID.txt 
4. run **export_InfluxDB_to_csv.sh** on **database server** to dump data from InfluxDB to .csv files. csv filename: nodeID.csv. 
5. run **scp command on database server** to push .csv files to GPU server (scp -p 22 csvfile user@130.245.191.166:path_to_save_data).
6. run **load_data_from_csv.py** on **GPU server** to convert .csv file to .npy (for baseband data) and .txt (for timestamps file). 
7. OR run **compute_doppler_from_UWB.py** to load csv data, segment file and plot range profile and doppler of each segment/activity.