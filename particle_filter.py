__author__ = "Erik Seetao"

import numpy as np 
from scipy.stats import multivariate_normal
import math

def mapCorrelation(im, x_im, y_im, vp, xs, ys): 
    '''
    Args:
        im: the map 
        x_im, y_im: physical x,y positions of the grid map cells
        vp[0:2,:]: occupied x,y positions from range sensor (in physical unit)  
        xs, ys: physical x,y,positions you want to evaluate "correlation" 
    Returns: 
        cpr: sum of the cell values of all the positions hit by range sensor

    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr

class particle_filter_SLAM:
    def __init__(self, num_particles, threshold):
        '''
        Initialize particles
        Args:
            num_particles: number of particles
            threshold: threshold for resampling condition

        '''
        self.thresh = threshold

        self.particle_weight = np.ones((num_particles,1)) / num_particles #normalization
        self.particles = np.zeros((num_particles , 3)) # 3dim for x y z


    def particle_prediction(self, disp, angle_change, variance = 0.000001): #try setting variance to 0.001
        '''
        Predict particles from encoder and IMU angles
        Args:
            disp: displacement
            angle_change: the change in angle
            variance: noise calculation
        Returns:
            Updated particles in x y z

        '''

        #for noise, x and y
        epsilon = np.random.randn(self.particles.shape[0], self.particles.shape[1]) #the noise
        
        if angle_change == 0:
            #no angle change
            self.particles[:,0] = self.particles[:,0] + disp * np.cos(self.particles[:,-1])
            self.particles[:,1] = self.particles[:,1] + disp * np.sin(self.particles[:,-1])
                
        else:
            #predict x and y, add into account angle
            xtheta = np.cos(self.particles[:,-1] + (angle_change/2))
            ytheta = np.sin(self.particles[:,-1] + (angle_change/2))
            self.particles[:,0] = self.particles[:,0] + (disp * np.sin(angle_change / 2) * xtheta / (angle_change / 2)) 
            self.particles[:,1] = self.particles[:,1] + (disp * np.sin(angle_change / 2) * ytheta / (angle_change / 2)) 
            #predict theta
            self.particles[:,2] = self.particles[:,2] + angle_change


        self.particles = self.particles + epsilon * variance #add in noise
        

    def particle_update(self, lidar, observation_model):
        '''
        Particle update for the filter weights from lidar 
        Args:
            lidar: lidar scans
            observation_model: observation from observation_model.py to load in occupancy map
        Returns:
            Updated particle weights
        '''

        map_data = observation_model.map_data
        occupancy_map = observation_model.occupancy_map 


        #set up for mapcorrelation
        x_im = np.arange(map_data["xmin"], map_data["xmax"] + map_data["res"], map_data["res"])
        y_im = np.arange(map_data["ymin"], map_data["ymax"] + map_data["res"], map_data["res"])

        p = np.zeros_like(self.particle_weight)

        total_particles = self.particles.shape[0]
        for index in range(total_particles):
            xs = self.particles[index][0] + np.arange(-0.05,0.1,0.05) #set up for mapcorrelation inputs
            ys = self.particles[index][1] + np.arange(-0.05,0.1,0.05) 


            #mapCorrelation updated, accounting for angle
            nx = occupancy_map.shape[0]
            ny = occupancy_map.shape[1]
            xmin = x_im[0]
            xmax = x_im[-1]
            xresolution = (xmax - xmin) / (nx - 1)

            ymin = y_im[0]
            ymax = y_im[-1]
            yresolution = (ymax - ymin) / (ny - 1)

            nxs = xs.size
            nys = ys.size
            cpr = np.zeros((nxs, nys))
            theta = self.particles[index][2]

            for jy in range(0, nys):
                y1 = lidar[0,:] * np.sin(theta) + lidar[1,:] * np.cos(theta) + ys[jy] 
                for jx in range(0, nxs):

                    x1 = lidar[0,:] * np.cos(theta) - lidar[1,:] * np.sin(theta) + xs[jx] 
                    y1 = lidar[0,:] * np.sin(theta) + lidar[1,:] * np.cos(theta) + ys[jy] 
                    ix = np.int16(np.round((x1 - xmin) / xresolution))
                    iy = np.int16(np.round((y1 - ymin) / yresolution))
                    cpr[jx, jy] = np.sum(occupancy_map[iy, ix])


            correlation = cpr

            p[index] = np.max(correlation)

            #convert to coord arrays
            location = np.unravel_index(correlation.argmax(), correlation.shape) 

            #For dead reckoning
            #self.particles[index,0] = self.particles[index,0] + (location[0] - 4) * map_data["res"] #x
            #self.particles[index,1] = self.particles[index,1] + (location[1] - 4) * map_data["res"] #y


            self.particles[index,0] = self.particles[index,0] + (location[0] - 1) * map_data["res"]
            self.particles[index,1] = self.particles[index,1] + (location[1] - 1) * map_data["res"]

        #set up for normalizing 
        p = np.exp(p - np.max(p))
        p = p / np.sum(p)            
        
        #finishing up
        self.particle_weight = self.particle_weight * p #new weight
        self.particle_weight = self.particle_weight / np.sum(self.particle_weight) #normalize

        #resampling condition
        if (1 / np.sum(np.square(self.particle_weight))) < self.thresh: #for use in stratified 
            self.resample()
        

    def resample(self): #recommend setting resample value to 0.01
        '''
        Resample when called upon by resample condition, using stratified resampling (low variance)
        Returns:
            new resampled particles
        '''

        j = 0 #initialize
        c = self.particle_weight[0] #load weight into c

        new_particles = self.particles.copy()
        number_particles = self.particles.shape[0]

        for k in range(number_particles):

            u = np.random.uniform(0, 1 / number_particles) #from U[0, 1/N] distribution
            b = u + (k / number_particles) # B = u + (k-1)/N, k already indexed one before

            while b > c:

                j = j + 1
                c = c + self.particle_weight[j] # c = c + alpha

            new_particles[k,:] = self.particles[j,:]

        #add (mu, 1/N) to the new set
        self.particles = new_particles
        self.particle_weight = np.ones((number_particles,1)) / number_particles

    @property
    def resample_condition(self):
        '''
        Resample if effective number of particles lower than threshold
        Returns:
            True/False if we need to resample or not

        '''
        return (1 / np.sum(np.square(self.particle_weight))) < self.thresh

    @property
    def highest_prob(self):
        '''
        Returns:
            Particle with the highest probability
            
        '''
        #this is the delta * alpha 
        return np.sum(self.particles * self.particle_weight, axis = 0)



