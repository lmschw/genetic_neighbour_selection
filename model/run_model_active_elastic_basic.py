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
    def __init__(self, domain_size, number_particles, radius, c_values, speed=1, alpha=0.1, beta=0.1, spring_constant=0.1, natural_distance=1, sensing_noise=0, actuation_noise=0):
        self.domain_size = np.array(domain_size)
        self.number_particles = number_particles
        self.radius = radius
        self.c_values = c_values
        self.speed = speed
        self.alpha = alpha
        self.beta = beta
        self.spring_constant = spring_constant
        self.natural_distance = natural_distance
        self.sensing_noise = sensing_noise
        self.actuation_noise = actuation_noise

    def initialize_state(self):
        positions = self.domain_size*np.random.rand(self.number_particles,len(self.domain_size))
        orientations = sorient.normalize_orientations(np.random.rand(self.number_particles, len(self.domain_size))-0.5)
        return positions, orientations
    
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
    
    def get_force(self, positions):
        neighbours = shelp.get_neighbours(positions=positions, domain_size=self.domain_size, radius=self.radius)
        neighbours = neighbours[:, :, np.newaxis] + neighbours[:, :, np.newaxis]
        neighbours = np.repeat(neighbours, 2, axis=2)
        distances = positions[:, np.newaxis] - positions[np.newaxis, :]
        distances = np.where((neighbours), distances, 0)
        spring_constant_div = -self.spring_constant/self.natural_distance
        distances_norm = np.linalg.norm(distances, axis=2, keepdims=True)
        distances_norm_inf = np.where(distances_norm == 0, np.inf, distances_norm)
        delta_distance = distances_norm - self.natural_distance
        norm_div = distances / distances_norm_inf
        force_components = spring_constant_div * delta_distance * norm_div
        c_values = np.repeat(self.c_values[np.newaxis, :, np.newaxis], 2, axis=2)
        force_components_c_values = force_components * c_values
        force = np.sum(force_components_c_values, axis=1)
        return force
    
    def update_positions(self, positions, orientations, forces):
        angles = sorient.compute_angles_for_orientations(orientations=orientations)
        unit_vector_heading = sorient.get_unit_vectors_for_angles(angles=angles)
        base = self.speed * unit_vector_heading
        chis_angles = np.random.random(size=self.number_particles) * 2 * np.pi
        chis = sorient.get_unit_vectors_for_angles(chis_angles)
        force_with_noise = forces + (self.sensing_noise * chis)
        new_pos = base + self.alpha * (force_with_noise * unit_vector_heading) * unit_vector_heading
        new_pos += positions
        new_pos += -self.domain_size*np.floor(new_pos/self.domain_size)
        return new_pos

    def update_orientations(self, positions, orientations, forces):
        unit_vector_heading_perpendicular = sorient.get_perpendicular_vectors(vectors=orientations)
        chis = np.random.normal(loc=1, scale=1, size=(self.number_particles, 2))
        force_with_noise = forces + (self.sensing_noise * chis)
        new_orients = self.beta * (force_with_noise * unit_vector_heading_perpendicular) + (self.actuation_noise * chis)
        new_orients = sorient.normalize_orientations(new_orients)
        return new_orients

    def prepare_simulation(self, initialState, dt, tmax):
        if any(ele is None for ele in initialState):
            positions, orientations = self.initialize_state()
        else:
            positions, orientations = initialState

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
        self.orientations_history = np.zeros((self.num_intervals,self.number_particles,len(self.domain_size)))

        self.positions_history[0,:,:] = positions
        self.orientations_history[0,:,:] = orientations

        return positions, orientations

    def simulate(self, initialState=(None, None, None), dt=None, tmax=None):
        positions, orientations = self.prepare_simulation(initialState=initialState, dt=dt, tmax=tmax)
        for t in range(0, self.tmax, self.dt):
            self.t = t

            forces = self.get_force(positions=positions)

            positions = self.update_positions(positions=positions, orientations=orientations, forces=forces)
            orientations = self.update_orientations(positions=positions, orientations=orientations, forces=forces)

            self.positions_history[t,:,:]=positions
            self.orientations_history[t,:]=orientations

        return (self.dt*np.arange(self.num_intervals), self.positions_history, self.orientations_history)
