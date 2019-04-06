__author__ = "Erik Seetao"


import numpy as np
import math

class encoder:
    def __init__(self, encoder_counts, time_stamps, meters_per_tick = 0.022):
        '''
        Class to process IMU measurements. Wheels are given as [FrontRight FrontLeft RearRight RearLeft]

        Args:
            encoder_counts: number of counts
            time_stamps: array of time stamps
        '''

        #take the average for front and rear sets
        left_dist = 0.0022 * (encoder_counts[0,:] + encoder_counts[2,:]) / 2 #FR FL
        right_dist = 0.0022 * (encoder_counts[1,:] + encoder_counts[3,:]) / 2 #RR RL

        self.average_displacement = (left_dist + right_dist) / 2 
        self.time_stamps = time_stamps


    def __getitem__(self,index):
        '''
        Synchronize time stamps and angular velocity 

        Args:
            index: index of data to extract
        '''

        return self.average_displacement[index]

    def __len__(self):
        '''
        Number of timestamps from encoder
        '''

        return self.time_stamps.shape[0]