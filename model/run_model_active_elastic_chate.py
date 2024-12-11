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

class ActiveElasticRunModel:
    """
    Performs the actual simulation of the agents interacting in the domain.
    """
    def __init__(self, domain_size, number_particles, radius, speed=1, mu=1.5, kappa=0.1, noise=0, rotational_noise=1):
        self.domain_size = np.array(domain_size)
        self.number_particles = number_particles
        self.radius = radius
        self.speed = speed
        self.mu = mu
        self.kappa = kappa
        self.noise = noise
        self.rotational_noise = rotational_noise

    def initialize_state(self):
        positions = self.domain_size*np.random.rand(self.number_particles,len(self.domain_size))
        polarities = sorient.normalize_angles(np.random.rand(self.number_particles) * 2 * np.pi)
        return positions, polarities
    
    def compute_bearings(self, positions):
        """
        Computes the bearings for every particle.

        Params:
            - positions (np.array): the position of every particle at the current timestep

        Returns:
            Numpy array containing the bearings of all particles. Values between [-pi, pi]
        """
        xDiffs = positions[:,0,np.newaxis]-positions[np.newaxis,:,0]
        yDiffs = positions[:,1,np.newaxis]-positions[np.newaxis,:,1]

        return np.arctan2(yDiffs, xDiffs)
    
    def update_positions(self, positions, polarities):
        unit_vectors_polarities = sorient.get_unit_vectors_for_angles(polarities)
        distances = sorient.get_differences_sqrt(array=positions, domain_size=self.domain_size)
        bearings = self.compute_bearings(positions=positions)
        unit_vectors_bearings = sorient.get_unit_vectors_for_angles(bearings)
        a = self.speed * unit_vectors_polarities
        dist = (self.radius-distances)
        combo = dist[:, None] * unit_vectors_bearings
        b = self.mu * np.sum(combo, axis=2)
        new_pos = a + b
        new_pos += -self.domain_size*np.floor(new_pos/self.domain_size)
        return new_pos

    def update_polarities(self, positions, polarities):
        neighbours = shelp.get_neighbours(positions=positions, domain_size=self.domain_size, radius=self.radius)
        distances = sorient.get_angle_differences(angles=polarities)
        sin_distances = np.sin(distances)
        sin_neighbours = np.where(neighbours, sin_distances, 0)
        polarities = self.kappa * np.sum(sin_neighbours, axis=1) + np.sqrt(2*self.rotational_noise)*self.noise
        return polarities

    def prepare_simulation(self, initialState, dt, tmax):
        if any(ele is None for ele in initialState):
            positions, polarities = self.initialize_state()
        else:
            positions, orientations = initialState
            polarities = sorient.compute_angles_for_orientations(orientations=orientations)

        if dt is None and tmax is not None:
            dt = 1
        
        if tmax is None:
            tmax = (10**3)*dt
            dt = np.average(10**(-2)*(np.max(self.domain_size)/self.speed))

        self.tmax = tmax
        self.dt = dt

        # Initialisations for the loop and the return variables
        self.num_intervals=int(tmax/dt+1)

        self.positions_history = np.zeros((self.num_intervals,self.number_particles,len(self.domain_size)))
        self.polarities_history = np.zeros((self.num_intervals,self.number_particles,len(self.domain_size)))

        self.positions_history[0,:,:] = positions
        self.polarities_history[0,:,:] = sorient.compute_uv_coordinates_for_list(angles=polarities)

        return positions, polarities

    def simulate(self, initialState=(None, None, None), dt=None, tmax=None):
        positions, polarities = self.prepare_simulation(initialState=initialState, dt=dt, tmax=tmax)
        for t in range(0, self.tmax, self.dt):
            self.t = t
            positions = self.update_positions(positions=positions, polarities=polarities)
            polarities = self.update_polarities(positions=positions, polarities=polarities)

            self.positions_history[t,:,:]=positions
            self.polarities_history[t,:]=sorient.compute_uv_coordinates_for_list(angles=polarities)

        return (self.dt*np.arange(self.num_intervals), self.positions_history, self.polarities_history)
