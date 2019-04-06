__author__ = "Erik Seetao"

import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt#; plt.ion()

import os

from lidar_processing import lidar
from imu_processing import imu
from encoder_processing import encoder

from observation_model import observation_model_SLAM
from particle_filter import particle_filter_SLAM


class SLAM:
    def __init__(self, particle_filter, observation_model):
        '''
        Initialize SLAM
        Args:
            particle_filter: particle filter algorithm
            observation_model: observation model

        '''

        self.particle_filter = particle_filter
        self.observation_model = observation_model

        
    def __call__(self, lidar, imu, encoder): #given all the sensor data
        '''
        Main SLAM function
        Args:
            lidar: lidar data
            imu: IMU data
            encoder: encoder data

        '''

        #initialize time stamps for all sensor data
        lidar_time = lidar.time_stamps
        imu_time = imu.time_stamps
        encoder_time = encoder.time_stamps

        x = []
        y = []

        for index in range(lidar_time.shape[0]):
            if index == 0: #first scan not much going on

                state = self.particle_filter.highest_prob
                occupancy_grid,_ =  self.observation_model.create_map(state, lidar[0]) #initialize the scan with the first instance of lidar

            else:
                #find valid indexes for IMU and encoder
                imu_valid = np.logical_and(imu_time >= lidar_time[index - 1], imu_time < lidar_time[index])

                if np.sum(imu_valid) == 0: #no angle change
                    angle_change = 0

                else:
                    lidar_time_disp = lidar_time[index] - lidar_time[index - 1]
                    angle_change = np.mean(imu[imu_valid]) * lidar_time_disp


                encoder_valid = np.logical_and(encoder_time >= lidar_time[index - 1], encoder_time < lidar_time[index])
                displacement = np.sum(encoder[encoder_valid])

                
                if displacement != 0 or angle_change != 0: #if robot is not moving, do not call particle filter

                    self.particle_filter.particle_prediction(displacement, angle_change, variance = 0.000001) 
                    self.particle_filter.particle_update(lidar[index], self.observation_model)

                #setup state with occupancy grid
                state = self.particle_filter.highest_prob
                occupancy_grid,_ =  self.observation_model.create_map(state,lidar[index]) #with lidar index

                #update
                #rounding down with np.floor
                map_res = self.observation_model.map_data["res"]
                x = x + [np.floor((state[0] - self.observation_model.minmatrix_plot[0]) / map_res).astype(np.uint16)]
                y = y + [np.floor((state[1] - self.observation_model.minmatrix_plot[1]) / map_res).astype(np.uint16)]
                
                print("Finished ", index, " iteration")

            if index % 100 == 0: #save per 100 iterations

                plt.scatter(x, y, s = 0.01, c = 'g')
                #plt.imshow(self.observation_model._trajectory_plot) #just plots trajectory model, no map
                plt.imshow(occupancy_grid, cmap='gray') #plots everything
                plt.savefig("/Users/eseetao/Documents/School Docs/ECE276A/Project 2/plots/{}.png".format(index))
                plt.close()

                

       

if __name__ == '__main__':

    dataset = 21
    data_path = "/Users/eseetao/Documents/School Docs/ECE276A/Project 2/data/"

    with np.load(data_path + "Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps

        encoder_data = encoder(encoder_counts,encoder_stamps)

    with np.load(data_path + "Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

        displacement = np.array([0,0.29833/2,0]) #displacement of lidar to body
        lidar_data = lidar(lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamps, displacement)

    with np.load(data_path + "Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

        imu_data = imu(imu_angular_velocity,imu_stamps)
    
    if dataset != 23: #dataset 23 doesn't have kinect data
        with np.load(data_path + "Kinect%d.npz"%dataset) as data:
            disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
            rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

    print("size of dataset ",dataset," is: ",lidar_ranges.shape)

    #load model
    observation_model = observation_model_SLAM(40, -40, 40, -40, 0.05, p_truepos = 0.80, p_trueneg = 0.80) 

    #initialize particle filter
    particle_filter_model = particle_filter_SLAM(100,5) #100 particles with threshold 5

    motion_model = SLAM(particle_filter_model, observation_model)
    motion_model(lidar_data, imu_data, encoder_data)
