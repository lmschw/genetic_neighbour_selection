import numpy as np
import random
import scipy.integrate as integrate
import csv
import matplotlib.pyplot as plt

from model.active_elastic.run_model_active_elastic_chate import ActiveElasticRunModel
from model.genetic_algorithm_model import GeneticAlgorithm
import services.service_preparation as sprep
import services.service_orientations as sorient
import services.service_logging as slog
import services.service_helper as shelp


class GeneticAlgorithmActiveElastic(GeneticAlgorithm):
    def __init__(self, radius, tmax, domain_size=(None, None), density=None, number_particles=None, speed=1, noise_percentage=0, 
                 num_generations=1000, num_iterations_per_individual=10, add_ranking_by=[True, True, True], add_own_orientation=False, add_random=False, 
                 use_norm=True, c_values_norm_factor=0, orientations_difference_threshold=2*np.pi, zero_choice_probability_initial=None,
                 zero_choice_probability_mutation=0, start_timestep_evaluation=0, changeover_point_timestep=0, start_order=None, 
                 target_order=1, population_size=100, bounds=[-1, 1], update_to_zero_bounds=[0,0], mutation_scale_factor=1, 
                 crossover_rate=0.5, early_stopping_after_gens=None, elite_size=2, sigma=0.1, introduce_new_values_probability=0,
                 events=None):
        """
        Models the GA approach.

        Params:
            - radius (int): the perception radius of the particles
            - tmax (int): the number of timesteps for each simulation
            - domain_size (tuple of floats) [optional]: the dimensions of the domain
            - density (float) [optional]: the density of the particles within the domain
            - number_particles (int) [optional]: how many particles are within the domain
            - speed (float) [optional, default=1]: how fast the particles move
            - noise_percentage (float) [optional, default=0]: how much environmental noise is present in the domain
            - num_generations (int) [optional, default=1000]: how many generations are generated and validated
            - num_iterations_per_individual (int) [optional, default=10]: how many times the simulation is run for every individual
            - add_own_orientation (boolean) [optional, default=False]: should the particle's own orientation be considered (added to weights and orientations)
            - add_random (boolean) [optional, default=False]: should a random value be considered (added to weights and orientations). Orientation value generated randomly at every timestep
            - start_timestep_evaluation (int) [optional, default=0]: the first timestep for which the difference between expected and actual result should be computed
            - changeover_point_timestep (int) [optional, default=0]: if we expect a change in the order, this indicated the timestep for that change
            - start_order (int: 0 or 1) [optional]: the order at the start. If this is not set, half the simulation runs are started with an ordered starting condition and half with a disordered starting condition
            - target_order (int: 0 or 1) [optional, default=1]: the expected order at the end
            - population_size (int) [optional, default=100]: how many individuals are generated per generation
            - bounds (list of 2 ints) [optional, default=[-1, 1]]: the bounds for the c_value generation
        """

        super().__init__(radius=radius, tmax=tmax, domain_size=domain_size, density=density, number_particles=number_particles,
                         speed=speed, noise_percentage=noise_percentage, num_generations=num_generations, 
                         num_iterations_per_individual=num_iterations_per_individual, add_ranking_by=add_ranking_by, 
                         add_own_orientation=add_own_orientation, add_random=add_random, use_norm=use_norm, 
                         c_values_norm_factor=c_values_norm_factor, orientations_difference_threshold=orientations_difference_threshold,
                         zero_choice_probability_initial=zero_choice_probability_initial, 
                         zero_choice_probability_mutation=zero_choice_probability_mutation, start_timestep_evaluation=start_timestep_evaluation,
                         changeover_point_timestep=changeover_point_timestep, start_order=start_order, target_order=target_order,
                         population_size=population_size, bounds=bounds, update_to_zero_bounds=update_to_zero_bounds, 
                         mutation_scale_factor=mutation_scale_factor, crossover_rate=crossover_rate, 
                         early_stopping_after_gens=early_stopping_after_gens, elite_size=elite_size, sigma=sigma, 
                         introduce_new_values_probability=introduce_new_values_probability, events=events)
        

    def create_run_model(self, c_values):
        return ActiveElasticRunModel(domain_size=self.domain_size,
                                radius=self.radius,
                                speed=self.speed,
                                number_particles=self.number_particles)