import pandas as pd
import numpy as np
from scipy import stats 

from model.run_model import RunModel
import services.service_orientations as sorient
import services.service_helper as shelp

class BinRunModel(RunModel):
    """
    Performs the actual simulation of the agents interacting in the domain.
    """
    def __init__(self, domain_size, radius, noise, speed, number_particles, c_values, bin_bounds,
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
        self.bin_bounds = bin_bounds
        self.normalise_bin_bounds()

        self.own_contribution_factor = c_values[-1]
        bin_size = len(self.bin_bounds)-1
        bin_shape = np.count_nonzero(self.add_ranking_by) * [bin_size]
        self.c_values = np.reshape(c_values[:-1], bin_shape)

    def normalise_bin_bounds(self):
        if self.bin_bounds[0] < 0:
            self.bin_bounds += -self.bin_bounds[0]
        elif self.bin_bounds[0] > 0:
            self.bin_bounds -= self.bin_bounds[0]
        if self.bin_bounds[-1] != 1:
            self.bin_bounds = self.bin_bounds/self.bin_bounds[-1]
        self.bin_bounds = np.round(self.bin_bounds,2)

    def get_parameter_summary(self):
        # TODO: add new params
        return super().get_parameter_summary()
    
    def find_bins(self, factors, max):
        factors_norm = np.absolute(factors) / max
        bins = np.full(factors_norm.shape, -1)
        for bin_idx in range(len(self.bin_bounds)-1):
            # the lower bound always belongs to the higher bin but the upper bound is also included to satisfy cases where the highest level is the max
            bins = np.where(((factors_norm >= self.bin_bounds[bin_idx]) & (factors_norm <= self.bin_bounds[bin_idx+1])), bin_idx, bins)
        return bins
    
    def get_bin_coefficients(self, orientation_differences, position_differences, bearings):

        bins_orientation_difference = self.find_bins(factors=orientation_differences, max=2*np.pi)
        bins_position_difference = self.find_bins(factors=position_differences, max=self.radius)
        bins_bearing = self.find_bins(factors=bearings, max=2*np.pi)

        if np.count_nonzero(self.add_ranking_by) == 3:
            vals = self.c_values[bins_orientation_difference, bins_position_difference, bins_bearing]
        elif self.add_ranking_by[0] and self.add_ranking_by[1]:
            vals = self.c_values[bins_orientation_difference, bins_position_difference]
        elif self.add_ranking_by[0] and self.add_ranking_by[2]:
            vals = self.c_values[bins_orientation_difference, bins_bearing]
        elif self.add_ranking_by[1] and self.add_ranking_by[2]:
            vals = self.c_values[bins_position_difference, bins_bearing]
        elif self.add_ranking_by[0]:
            vals = self.c_values[bins_orientation_difference]
        elif self.add_ranking_by[1]:
            vals = self.c_values[bins_position_difference]
        else:
            vals = self.c_values[bins_bearing]
        
        # if the position is further away than the radius, it will not fit into any bin and therefore will have -1
        # those cases are set to 0 to make sure they are ignored. Such a correction is not needed for orientations
        # and bearings because these are circular. However, once a field of vision is included, TODO bearings will
        # need to be checked
        radius_corrected = np.where((bins_position_difference == -1), 0, vals)

        return radius_corrected


    def compute_new_orientations(self, positions, orientations):
        orientation_differences = sorient.get_differences(orientations, self.domain_size)
        position_differences = sorient.get_differences_sqrt(positions, self.domain_size)
        bearings = self.compute_bearings(positions=positions)

        # multiply each factor for each individual
        factors_overall = self.get_bin_coefficients(orientation_differences=orientation_differences,
                                                    position_differences=position_differences,
                                                    bearings=bearings)
        # normalise for each individual
        factors_overall_normalised = shelp.normalise(factors_overall, norm='l1')

        angles = sorient.compute_angles_for_orientations(orientations=orientations)
        angles_enhanced = np.array(self.number_particles * [angles])
        new_angles = np.sum(np.tensordot(angles_enhanced, factors_overall_normalised, axes=1), axis=1)
        # TODO figure out best bounds for Gaussian & look up other function generation methods
        return self.own_contribution_factor * orientations + (1-self.own_contribution_factor) * sorient.compute_uv_coordinates_for_list(angles=new_angles)
