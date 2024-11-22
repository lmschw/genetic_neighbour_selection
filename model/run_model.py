import pandas as pd
import numpy as np

import services.service_orientations as sorient

class RunModel:
    """
    Performs the actual simulation of the agents interacting in the domain.
    """
    def __init__(self, domain_size, radius, noise, speed, number_particles, c_values, 
                 add_ranking_by=[True, True, True], add_own_orientation=False, add_random=False, events=None):
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
        self.domain_size = np.array(domain_size)
        self.radius = radius
        self.noise = noise
        self.speed = speed
        self.number_particles = number_particles
        self.c_values = c_values
        self.add_ranking_by = add_ranking_by
        self.rank_by_orientation = add_ranking_by[0]
        self.rank_by_distance = add_ranking_by[1]
        self.rank_by_bearing = add_ranking_by[2]
        self.add_own_orientation = add_own_orientation
        self.add_random = add_random
        self.events = events

    def get_parameter_summary(self):
        return {"domain_size": self.domain_size.tolist(),
                "radius": self.radius,
                "noise": self.noise,
                "speed": self.speed,
                "number_particles": self.number_particles,
                "c_values": self.c_values.tolist(),
                "add_ranking_by": self.add_ranking_by,
                "add_own_orientation": self.add_own_orientation,
                "add_random": self.add_random}
        
    def initialize_state(self):
        """
        Initialises the state of the swarm at the start of the simulation.

        Params:
            None
        
        Returns:
            Arrays of positions and orientations containing values for every individual within the system
        """
        positions = self.domain_size*np.random.rand(self.number_particles,len(self.domain_size))
        orientations = sorient.normalize_orientations(np.random.rand(self.number_particles, len(self.domain_size))-0.5)

        return positions, orientations
    
    def generate_noise(self):
        """
        Generates some noise based on the noise amplitude set at creation.

        Params:
            None

        Returns:
            An array with the noise to be added to each individual
        """
        return np.random.normal(scale=self.noise, size=(self.number_particles, len(self.domain_size)))
    
    def create_sorted_orientations_array(self, positions, orientations):
        """
        Sorts the orientations by orientation differences, distances and bearings. Then combines these three rankings into a single array and adds the particle's
        own orientation and a random orientation as needed.

        Params:
            - positions (np.array): the position of every particle at the current timestep
            - orientations (np.array): the orientation of every particle at the current timestep

        Returns:
            A numpy array containing the orientations of all particles to correspond to the c_values.
        """

        # we remove the diagonal after the sorting to remove the particle's own information
        diagonal_mask = np.full((self.number_particles, self.number_particles), False)
        np.fill_diagonal(diagonal_mask, True)

        orients = []
        if self.rank_by_orientation == True:
            orientation_differences = np.ma.MaskedArray(sorient.get_differences(orientations, self.domain_size), mask=diagonal_mask)
            orientation_diff_sorted_indices = np.argsort(orientation_differences, axis=1)
            orientation_diff_sorted_indices_without_diagonal = orientation_diff_sorted_indices[:, :-1]
            orientations_by_orientation_diff = np.array(orientations[orientation_diff_sorted_indices_without_diagonal])
            orients = orientations_by_orientation_diff

        if self.rank_by_distance == True:
            distances = np.ma.MaskedArray(sorient.get_differences(positions, self.domain_size), mask=diagonal_mask)
            distances_sorted_indices = np.argsort(distances, axis=1)
            distances_sorted_indices_without_diagonal = distances_sorted_indices[:, :-1]
            orientations_by_distance = np.array(orientations[distances_sorted_indices_without_diagonal])
            if len(orients) == 0:
                orients = orientations_by_distance
            else:
                orients = np.append(orients, orientations_by_distance, axis=1)

        if self.rank_by_bearing == True:
            bearings = np.ma.MaskedArray(self.compute_bearings(positions=positions), mask=diagonal_mask)
            bearings_sorted_indices = np.argsort(bearings, axis=1)
            bearings_sorted_indices_without_diagonal = bearings_sorted_indices[:, :-1]
            orientations_by_bearing = np.array(orientations[bearings_sorted_indices_without_diagonal])
            if len(orients) == 0:
                orients = orientations_by_bearing
            else:
                orients = np.append(orients, orientations_by_bearing, axis=1)
                
        if self.add_own_orientation == True:
            orients = np.append(orients, orientations[:, np.newaxis, :], axis=1)
        if self.add_random == True:
            rands = np.random.uniform(low=-1, high=1, size=(self.number_particles, 2))
            orients = np.append(orients, rands[:, np.newaxis, :], axis=1)
        return orients
    
    def compute_new_orientations(self, positions, orientations):
        """
        Updates the orientations in accordance with the c_values.

        Params:
            - positions (np.array): the position of every particle at the current timestep
            - orientations (np.array): the orientation of every particle at the current timestep

        Returns:
            A numpy array containing the new orientation of every particle.
        """
        sorted_orientations = self.create_sorted_orientations_array(positions=positions, orientations=orientations)
        applied_orientations = np.tensordot(sorted_orientations, self.c_values, axes=(1, 0))

        return sorient.normalize_orientations(applied_orientations)
    
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
    
    def prepare_simulation(self, initialState, dt, tmax):
        """
        Prepares the simulation by initialising all necessary properties.

        Params:
            - initialState (tuple of arrays) [optional]: A tuple containing the initial positions of all particles, their initial orientations and their initial switch type values
            - dt (int) [optional]: time step
            - tmax (int) [optional]: the total number of time steps of the experiment

        Returns:
            Arrays containing the positions, orientations, neighbour selection mechanisms, ks, speeds and time delays.
        """
         # Preparations and setting of parameters if they are not passed to the method
        
        if any(ele is None for ele in initialState):
            positions, orientations = self.initialize_state()
        else:
            positions, orientations = initialState

        #print(f"t=pre, order={ServiceMetric.computeGlobalOrder(orientations)}")

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

        self.positions_history[0,:,:]=positions
        self.orientations_history[0,:,:]=orientations

        return positions, orientations

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

            orientations = self.compute_new_orientations(positions, orientations)
            orientations = sorient.normalize_orientations(orientations+self.generate_noise())

            positions += self.dt*(orientations * self.speed)
            positions += -self.domain_size*np.floor(positions/self.domain_size)

            self.positions_history[t,:,:]=positions
            self.orientations_history[t,:,:]=orientations

            # if t % 500 == 0:
            #     print(f"t={t}, order={sorient.compute_global_order(orientations)}")

        return (self.dt*np.arange(self.num_intervals), self.positions_history, self.orientations_history)
