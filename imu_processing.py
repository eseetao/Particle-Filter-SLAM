__author__ = "Erik Seetao"


import numpy as np
import math

from scipy.signal import butter, lfilter #to denoise imu data with lowpass

#do the same thing as lidar_processing.py
class imu:
    def __init__(self, angular_velocity, time_stamps):
        '''
        Process IMU to load into main SLAM file
        Args:
            angular_velocity: angular velocity per time stamp
            time_stamps: time stamps

        '''

        self.time_stamps = time_stamps

        freq_crit = 10
        freq_sample = 1 / (time_stamps[1] - time_stamps[0])

        #use butterworth filter for lowpass filter
        num, den = butter(1, freq_crit / (freq_sample / 2), btype = 'low') #use a first order filter with btype lowpass

        self.angle = angular_velocity[2,:]
        self.angle = lfilter(num, den, self.angle, axis = 0)


    def __getitem__(self, index):
        '''
        Synchronize time stamps and angular velocity 

        Args:
            index: index of data to extract
        '''

        return self.angle[index]

    def __len__(self):
        '''
        Number of timestamps from encoder
        '''
        
        return self.time_stamps.shape[0]