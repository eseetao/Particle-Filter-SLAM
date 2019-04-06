__author__ = "Erik Seetao"

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2


def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
        (sx, sy): start point of ray
        (ex, ey): end point of ray
    Returns:
        stacked np array of bresenham rays
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y))


class observation_model_SLAM:
    def __init__(self, xmax, xmin, ymax, ymin, res, p_truepos = 0.75, p_trueneg = 0.75): 
        '''
        Initialize observation model with log odds and occupancy map
        Args:
            xmax: positive limit of map in x axis
            xmin: negative limit of map in x axis
            ymax: positive limit of map in y axis
            ymin: negative limit of map in y axis
            res: resolution per unit

        ''' 
        self._p_truepos = p_truepos
        self._p_falsepos = 1 - p_truepos
        self._map_data = {"xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, "res":res} #map sizes, res from TA 

        #set up log odds update
        self._odds_update = np.log(p_truepos / (1 - p_truepos))
        self._shape = (int((self._map_data["xmax"] - self._map_data["xmin"]) // self._map_data["res"]) + 1, int((self._map_data["ymax"] - self._map_data["ymin"]) // self._map_data["res"]) + 1)


        self._minmatrix_plot = np.array([self._map_data['xmin'],self._map_data['ymin']]) #final plot to send to main with trajectory
        self._trajectory_plot = np.zeros(self._shape,dtype=np.float64) #testing for trajectory plot


        self.occupancy_map = np.zeros(self._shape,dtype=np.uint8)
        self.logodds_map = np.zeros(self._shape,dtype=np.float64)

    def create_map(self, particle, lidar_data):
        '''
        Create log odds map with bresenham ray updates per scan
        Args:
            particle: x y position of particles
            lidar_data: lidar scans
        Returns:
            self.occupancy_map: updated occupancy map
            self.logodds_map: updated log odds map

        '''

        body2world = np.array([[np.cos(particle[-1]), -np.sin(particle[-1]), 0, particle[0]], [np.sin(particle[-1]), np.cos(particle[-1]), 0, particle[1]], [0, 0, 1, 0], [0, 0, 0, 1]])

        #aggregate everything into this
        map_coordinates = np.floor((np.dot(body2world,lidar_data)[:2,:] - self._minmatrix_plot.reshape(-1,1)) / self._map_data["res"]).astype(np.uint16)#Transfer points into world co ordinates


        #setup rays for bresenham
        ray_start = np.floor((np.array([particle[0],particle[1]]) - self._minmatrix_plot) / self._map_data["res"]).astype(np.uint16)
        ray_end = map_coordinates.T.tolist()

        for ray in ray_end:
            scans = bresenham2D(ray_start[0], ray_start[1], ray[0], ray[1]).astype(np.uint16)

            #update existing log odds
            self.logodds_map[scans[1][-1],scans[0][-1]] = self.logodds_map[scans[1][-1], scans[0][-1]] + self._odds_update
            self.logodds_map[scans[1][1:-1],scans[0][1:-1]] = self.logodds_map[scans[1][1:-1], scans[0][1:-1]] - self._odds_update

        self.logodds_map[scans[1][0],scans[0][0]] = self.logodds_map[scans[1][0],scans[0][0]] - self._odds_update


        threshold = 0.95
        P_occupied = 1 / (1 + np.exp(-self.logodds_map))

        #update occupancy map based on thresholding
        self.occupancy_map = (P_occupied < 0.005) * (-1) + (P_occupied >= threshold) * (1) #yeah the *1 isn't necessary but you get the workflow
        
        
        return self.occupancy_map,self.logodds_map


    @property
    def p_truepos(self):
        '''
        Returns:
            p_truepos: probability of grid cell being occupied when it is measured as occupied

        '''
        #default to .75, but set to .80
        return self._p_truepos
    

    @property
    def minmatrix_plot(self):
        '''
        Returns:
            minmatrix_plot: statistics of plot

        '''
        #returning grid min for shifts, comes in handy in calling 
        return self._minmatrix_plot

    @property
    def trajectory_plot(self):
        '''
        Returns:
            trajectory_plot: plot of trajectory

        '''
        return self._trajectory_plot

    @property
    def map_data(self):
        '''
        Returns:
            map_data: dict statistics of map

        '''
        return self._map_data

    @property
    def shape(self):
        '''
        Returns:
            shape: shape of occupancy grid

        '''
        return self._shape
