import pandas as pd
import numpy as np
from scipy import stats 

from model.run_model import RunModel
import services.service_orientations as sorient
import services.service_helper as shelp

class DistributionRunModel(RunModel):
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
        super().__init__(domain_size=domain_size, radius=radius, noise=noise, speed=speed, number_particles=number_particles,
                         c_values=c_values, add_ranking_by=add_ranking_by, add_own_orientation=add_own_orientation, 
                         add_random=add_random, events=events)

    def get_parameter_summary(self):
        # TODO: add new params
        return super().get_parameter_summary()
    
    def get_factors(self, mean, standard_deviation, values):
        # Gaussian curve evaluation for all the values in the array
        distribution = stats.norm(loc=mean, scale=standard_deviation)
        return np.array(distribution.pdf(x=values))
    
    def compute_new_orientations(self, positions, orientations):
        orientation_differences = sorient.get_differences(orientations, self.domain_size)
        position_differences = sorient.get_differences(positions, self.domain_size)
        bearings = self.compute_bearings(positions=positions)

        factors_orientation_difference = self.get_factors(self.c_values[0], self.c_values[1], orientation_differences)
        factors_position_difference = self.get_factors(self.c_values[2], self.c_values[3], position_differences)
        factors_bearing = self.get_factors(self.c_values[4], self.c_values[5], bearings)

        # multiply each factor for each individual
        factors_overall = factors_orientation_difference * factors_position_difference * factors_bearing
        # normalise for each individual
        factors_overall_normalised = shelp.normalise(factors_overall, norm='l1')

        angles = sorient.compute_angles_for_orientations(orientations=orientations)
        new_angles = angles * factors_overall_normalised
        # TODO figure out best bounds for Gaussian & look up other function generation methods
        return self.c_values[6] * orientations + (1-self.c_values[6]) * sorient.compute_uv_coordinates_for_list(angles=new_angles)
