# Human-Data-Utils

## Install

To install hmdata:
```shell
$ pip install hmdata
```

## List of Data

* [TotalCapture](#TotalCapture)

## TotalCapture

The TotalCapture dataset is designed for 3D pose estimation from markerless multi-camera capture, It is the first dataset to have fully synchronised muli-view video, IMU and Vicon labelling for a large number of frames (∼1.9M), for many subjects, activities and viewpoints.

Please check the information on this page: [TotalCapture Dataset](https://cvssp.org/data/totalcapture/)

### Data Preparation

To request access to the TotalCapture Dataset, or for other queries please contact: a.gilbert@surrey.ac.uk.

Download the TotalCapture Dataset and prepare in the following format:
```
datafolder/
    |-> S1/
        |-> gyro_mag/
            |-> acting1_Xsens_AuxFields.sensors
            |-> acting2_Xsens_AuxFields.sensors
            |-> ...
        |-> imu/
            |-> s2_acting1_calib_imu_bone.txt
            |-> s2_acting1_calib_imu_ref.txt
            |-> s2_acting1_Xsens.sensors
            |-> ...
        |-> video/
            |-> acting1/
                |-> TC_S2_acting1_cam1.mp4
                |-> ...
                |-> TC_S2_acting1_cam8.mp4
            |-> acting2/
            |-> .../
            |-> walking3/
        |-> vicon/
            |-> acting1/
                |-> gt_skel_gbl_pos.txt
                |-> gt_skel_gbl_ori.txt
            |-> ...
    |->.../
    |->S5/
```

### Data Preprocess

Convert all videos to images:
```shell
$ cd example
$ bash video2images.sh data_dir num_of_processes
```
<!-- bash video2images.sh /media/ywj/Data/totalcapture/totalcapture 4 -->
<!-- bash video2images.sh /mnt/md0/yinw/project/data/totalcapture 4 -->