import pandas as pd
import numpy as np
from scipy import stats


import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, circmean, circvar
import pickle
from datetime import datetime

from model.run_model import RunModel
import services.service_orientations as sorient
import services.service_helper as shelp

class ActiveElasticRunModel(RunModel):
    """
    Performs the actual simulation of the agents interacting in the domain.
    """
    def __init__(self, domain_size, radius, noise, speed, number_particles, c_values, 
                 epsilon=12, sigma=0.7, alpha=2.0, beta=0.5, uc=0.05, umax=0.1, wmax=np.pi/2,
                 k1=0.6, k2=0.05, add_ranking_by=[True, True, True], add_own_orientation=False, 
                 add_random=False, events=None):
        """
        Params:
            - domain_size (tuple of floats) [optional]: the dimensions of the domain
            - radius (int): the perception radius of the particles
            - noise (float): the amount of environmental noise in the domain
            - speed (float): how fast the particles move
            - number_particles (int): how many particles are within the domain
            - c_values (np.array): the weights for the orientations
            - add_own_orientation (boolean) [optional, default=False]: should the particle's own orientation be considered (added to weights and orientations)
            - add_random (boolean) [optional, default=False]: should a random value be considered (added to weights and orientations). Orientation value generated randomly at every timestep
        """
        super().__init__(domain_size=domain_size, radius=radius, noise=noise, speed=speed, number_particles=number_particles,
                         c_values=c_values, add_ranking_by=add_ranking_by, add_own_orientation=add_own_orientation, 
                         add_random=add_random, events=events)
        
        self.epsilon = epsilon
        self.sigma = sigma
        self.sigmas = np.full(self.number_particles, self.sigma)
        self.alpha = alpha
        self.beta = beta
        self.uc = uc
        self.umax = umax
        self.wmax = wmax
        self.k1 = k1
        self.k2 = k2

    def get_parameter_summary(self):
        # TODO: add new params
        return super().get_parameter_summary()
    
    def compute_new_positions_and_orientations(self, positions, orientations):

        headings = sorient.compute_angles_for_orientations(orientations=orientations)
        ranked_headings = sorient.compute_angles_for_orientations(self.create_sorted_orientations_array(positions=positions, orientations=orientations))

        # Calculate forces
        # TODO add positions and bearings with their own c_values
        f_x, f_y = self.compute_fi(positions=positions, headings=headings, ranked_headings=ranked_headings)
        u, w = self.compute_u_w(f_x, f_y)

        # Project to local frame
        x_vel = np.multiply(u, np.cos(headings))
        y_vel = np.multiply(u, np.sin(headings))
        # print(f"X add: {x_vel}")
        # print(f"Y add: {y_vel}")

        # Update agents
        x = positions[:, 0] + x_vel * self.dt
        y = positions[:, 1] + y_vel * self.dt
        positions = np.column_stack([x,y])

        angles = self.wrap_to_pi(headings + w * self.dt)
        orientations = sorient.compute_uv_coordinates_for_list(angles=angles)

        return positions, orientations


    def compute_distances_and_angles(self, positions, headings):
        """
        Computes and returns the distances and its x and y elements for all pairs of agents

        """
        # Build meshgrid 
        pos_xs = positions[:, 0]
        pos_ys = positions[:, 1]
        xx1, xx2 = np.meshgrid(pos_xs, pos_xs)
        yy1, yy2 = np.meshgrid(pos_ys, pos_ys)

        # Calculate distances
        x_diffs = xx1 - xx2
        y_diffs = yy1 - yy2
        distances = np.sqrt(np.multiply(x_diffs, x_diffs) + np.multiply(y_diffs, y_diffs))  
        distances[distances > self.radius] = np.inf
        distances[distances == 0.0] = np.inf
        # print(f"Dists: {distances}")
        

        # Calculate angles in the local frame of reference
        angles = np.arctan2(y_diffs, x_diffs) - headings[:, np.newaxis]
        # print(f"Angles: {angles}")

        return distances, angles

    def get_position_elements(self, distances, angles):
        """
        Calculates the x and y components of the proximal control vector

        """  
        forces = -self.epsilon * (2 * (self.sigmas[:, np.newaxis] ** 4 / distances ** 5) - (self.sigmas[:, np.newaxis] ** 2 / distances ** 3))
        forces[distances == np.inf] = 0.0
        # print(f"Forces: {forces}")


        p_x = np.sum(np.multiply(forces, np.cos(angles)), axis=1)
        p_y = np.sum(np.multiply(forces, np.sin(angles)), axis=1)

        return p_x, p_y

    def get_heading_alignment_elements(self, headings, ranked_headings):
        """
        Calculates the x and y components of the heading alignment vector

        """  
        # All this is doing is getting the vectorial avg of the headings
        # TODO limit by c_values
        alignment_coss = np.sum(self.c_values * np.cos(ranked_headings), axis=1)
        alignment_sins = np.sum(self.c_values * np.sin(ranked_headings), axis=1)
        alignment_angs = np.arctan2(alignment_sins, alignment_coss)
        alignment_mags = np.sqrt(alignment_coss**2 + alignment_sins**2)

        h_x = alignment_mags * np.cos(alignment_angs - headings)
        h_y = alignment_mags * np.sin(alignment_angs - headings)

        return h_x, h_y

    def compute_fi(self, positions, headings, ranked_headings):
        """
        Computes the virtual force vector components

        """  
        dists, angles = self.compute_distances_and_angles(positions=positions, headings=headings)

        p_x, p_y = self.get_position_elements(dists, angles)
        h_x, h_y = self.get_heading_alignment_elements(headings=headings, ranked_headings=ranked_headings)

        f_x = self.alpha * p_x + self.beta * h_x 
        f_y = self.alpha * p_y + self.beta * h_y 
        # print(f"Fx: {f_x}")
        # print(f"Fy: {f_y}")
        

        return f_x, f_y
    
    def compute_u_w(self, f_x, f_y):
        """
        Computes u and w given the components of Fi

        """
        u = self.k1 * f_x + self.uc
        u[u > self.umax] = self.umax
        u[u < 0] = 0.0

        w = self.k2 * f_y
        w[w > self.wmax] = self.wmax
        w[w < -self.wmax] = -self.wmax

        return u, w


    def wrap_to_pi(self, x):
        """
        Wrapes the angles to [-pi, pi]

        """
        x = x % (np.pi * 2)
        x = (x + (np.pi * 2)) % (np.pi * 2)

        x[x > np.pi] = x[x > np.pi] - (np.pi * 2)

        return x
    
    def simulate(self, initialState=(None, None, None), dt=None, tmax=None):
        """
        Runs the simulation experiment.
        First the parameters are computed if they are not passed. 
        Then the positions and orientations are computed for each particle at each time step.

        Params:
            - initialState (tuple of arrays) [optional]: A tuple containing the initial positions of all particles and their initial orientations
            - dt (int) [optional]: time step
            - tmax (int) [optional]: the total number of time steps of the experiment

        Returns:
            (times, positions_history, orientations_history)
        """
       
        positions, orientations = self.prepare_simulation(initialState=initialState, dt=dt, tmax=tmax)
        #print(f"t=start, order={sorient.compute_global_order(orientations)}")
        
        for t in range(self.num_intervals):
            self.t = t
            # if t % 5000 == 0:
            #     print(f"t={t}/{self.tmax}")

            if self.events != None:
                for event in self.events:
                    orientations = event.check(self.number_particles, t, positions, orientations)

            positions, orientations = self.compute_new_positions_and_orientations(positions, orientations)
            orientations = sorient.normalize_orientations(orientations+self.generate_noise())
            positions += -self.domain_size*np.floor(positions/self.domain_size)

            self.positions_history[t,:,:]=positions
            self.orientations_history[t,:,:]=orientations

            # if t % 500 == 0:
            #     print(f"t={t}, order={sorient.compute_global_order(orientations)}")

        return (self.dt*np.arange(self.num_intervals), self.positions_history, self.orientations_history)
