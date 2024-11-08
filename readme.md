##
Not updated since 2025-11-8. Move to ADL_Detection.

## Time Alignment for ADL data in 2023
Video and timestamp for each video frame is on BoX. You can NOT download video from BoX. You can download the rgb_ts.txt file.
### Align Android app to camera
#### Get anchor time from video
- watch the video on HAR2, get the time of starting "sit on the couch and watch the video on the monitor [2 min]. (Click "Start"!)". Usually you can see clearly the person clicks on the phone before stiing on the couch.
- record the time of the action in video file, (e.g. 5 minuets 6 second). Calculate the video frame index. video FPS=24. (e.g. (5*60+6)*24=7344).
- read the corresponding timestamp of the video frame from rgb_ts.txt. (e.g. in the 7244th line. convert timestamp to datetime format.). Denote by Tv.
#### Get anchor time from app
- read start time ar the 11th line of the segmentation file, which is the timestamp recorded on the phone while starting "sit on the couch and watch the video on the monitor [2 min]. (Click "Start"!)". Denote by Ta.
#### Calculate time shift
- Tv - Ta

### Correct the segmentation file using (Tv - Ta)
- run time_shift() in manual_segment.py. input the time shift (Tv - Ta).
- *_shifted.txt will be saved under the same segmentation path.

### Align sensor to camera
#### Get anchor time from video
- watch the video from HAR2, record the timestamp when the person gets nearest the sensor, i.e. walking towards couch and standing there for 1~2 seconds.
- get the timestamp of the action in the same way. Denote by Tv2.
#### Get anchor time from sensor
- Generate range doppler video from baseband data. Run plot_range_doppler_before_calibration.py. 
- watch the doppler video, record the time in doppler video when you see the person walking towards the sensor to the nearest point. e.g. (2 minutes 3 seconds)
  - Since we use shifted segmentation which is aligned with video (not aligned with sensor yet), the ground truth may be slightly time shifted from the pattern you observed in Doppler video.
- calculate doppler frame index (e.g. (2*60+3)*24=2952). calculate corresponding baseband data frame index. (e.g. 2952*5=14760)
- read the timestamp file of uwb sensor data. (e.g. at the 14760th line.). Denote by Ts.
#### Calculate time shif
- Tv2 - Ts
- record the time shift in a file. e.g. {session_id}_{sensor_idx}_timeshift.npy. [2, 3]

## This repository includes
1. ADL data processing, data integrity checking
2. finetune ViT in multiple ways using M4X dataset: finetune last layer, fully finetune, LoRA, w/o dropout layers on CNN/FF layers
3. train a ViT from scratch using M4X dataset
4. reprogram llama for M4X dataset
5. CoT a llava model (not finished)

## Packages
model (for llama from huggingface), transformers (for ViT)

## Segment and load data from InfluxDB
1. watch video on Box and record segmentation information in file.
2. Download **rgb_ts.txt** from Box. 
3. run **manual_segment.py** to get start and stop datatime of each activity. Save segment result in nodeID-DataSessionID.txt 
4. run **export_InfluxDB_to_csv.sh** on **database server** to dump data from InfluxDB to .csv files. csv filename: nodeID.csv. 
5. run **scp command on database server** to push .csv files to GPU server (scp -p 22 csvfile user@130.245.191.166:path_to_save_data).
6. run **load_data_from_csv.py** on **GPU server** to convert .csv file to .npy (for baseband data) and .txt (for timestamps file). 
7. OR run **compute_doppler_from_UWB.py** to load csv data, segment file and plot range profile and doppler of each segment/activity.
