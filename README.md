# Particle-Filter-SLAM

6 Files are attached in this project:

SLAM_main.py
particle_filter.py
observation_model.py
encoder_processing.py
lidar_processing.py
imu_processing.py

The encoder, lidar, and IMU processing files are used to read the data from the robot. The particle_filter contains the implementation of the particle filter while the observation_model contains the occupancy map and logodds map. 
These files are called in SLAM_main which contains all of necessary implementations for a particle filter SLAM.

An example of the SLAM map is given below:
![alt text](https://github.com/eseetao/Particle-Filter-SLAM/blob/master/Images/Data20dead2.png)

Author: Erik Seetao
