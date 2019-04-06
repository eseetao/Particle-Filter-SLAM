__author__ = "Erik Seetao"


import numpy as np
import math
   
class lidar:
    def __init__(self, lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamps, displacement):
        '''
        Process lidar data to load to main SLAM file, given from Hokuyo LIDAR
        Args:
            lidar_angle_min: [rad] start angle of scan
            lidar_angle_max: [rad] end angle of scan
            lidar_angle_increment: [rad] angular distance increments between measurements
            lidar_range_min: [m] minimum range 
            lidar_range_max: [m] maximum range
            lidar_ranges: range data for lidar
            lidar_stamps: timestamps of each lidar scan
            displacement: displacement from lidar to body

        '''

        angles = np.arange(lidar_angle_min, lidar_angle_max + lidar_angle_increment, lidar_angle_increment, dtype=np.float64)[:1081]

        range_max = lidar_range_max
        range_min = lidar_range_min
        
        self.ranges = lidar_ranges
        self.time_stamps = lidar_stamps

        self._lidar_stats = {"angle_span":angles, "range_max":range_max, "range_min":range_min, "displacement":displacement}

    @property
    def lidar_stats(self):
        '''
        Gets dictionary of lidar ranges, angles, and displacement
        '''

        return self._lidar_stats
    
    def __getitem__(self,index):
        '''
        Iterator for laser scan ranges and time stamps
        Args:
            index: index of data to extract
        Returns:
            homoge_coord: homogeneous coordinates of the x y polar coordinates into the world frame

        '''

        ranges = self.ranges[:,index]
        indValid = np.logical_and((ranges <= self._lidar_stats["range_max"]),(ranges >= self._lidar_stats["range_min"]))
        
        ranges = ranges[indValid]
        angles = self._lidar_stats["angle_span"][indValid] 

        #incorporate polar to cartesian with the displacement from lidar to body
        x = ranges * np.cos(angles) + self._lidar_stats["displacement"][0]
        y = ranges * np.sin(angles) + self._lidar_stats["displacement"][1]
        homoge_coord = np.stack([x, y,np.zeros_like(x),np.ones_like(x)], axis=0)
        return homoge_coord

    def __len__(self):
        '''
        Number of timestamps from encoder
        '''

        return self.time_stamps.shape[0]