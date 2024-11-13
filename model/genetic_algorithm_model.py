import numpy as np
import random
import scipy.integrate as integrate
import csv
import matplotlib.pyplot as plt

from model.run_model import RunModel
import services.service_preparation as sprep
import services.service_orientations as sorient
import services.service_logging as slog
import services.service_helper as shelp


class GeneticAlgorithm:
    def __init__(self, radius, tmax, domain_size=(None, None), density=None, number_particles=None, speed=1, noise_percentage=0, 
                 num_generations=1000, num_iterations_per_individual=10, add_own_orientation=False, add_random=False, 
                 use_norm=True, c_values_norm_factor=0, orientations_difference_threshold=2*np.pi, zero_choice_probability_initial=None,
                 zero_choice_probability_mutation=0, start_timestep_evaluation=0, changeover_point_timestep=0, start_order=None, 
                 target_order=1, population_size=100, bounds=[-1, 1], update_to_zero_bounds=[0,0], mutation_scale_factor=1, 
                 crossover_rate=0.5, early_stopping_after_gens=None, elite_size=2, sigma=0.1, introduce_new_values_probability=0):
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

        self.radius = radius
        self.num_generations = num_generations
        self.num_iterations_per_individual = num_iterations_per_individual
        self.speed = speed
        self.noise_percentage = noise_percentage
        self.noise = sprep.get_noise_amplitude_value_for_percentage(self.noise_percentage)

        self.tmax = tmax
        self.add_own_orientation = add_own_orientation
        self.add_random = add_random
        self.use_norm = use_norm
        self.c_values_norm_factor = c_values_norm_factor
        self.orientations_difference_threshold = orientations_difference_threshold
        self.zero_choice_probability_initial = zero_choice_probability_initial
        self.zero_choice_probability_mutation = zero_choice_probability_mutation
        self.start_timestep_evaluation = start_timestep_evaluation
        self.changeover_point_timestep = changeover_point_timestep
        self.start_order = start_order
        self.target_order = target_order
        self.population_size = population_size
        self.bounds = bounds
        self.update_to_zero_bounds = update_to_zero_bounds
        self.mutation_scale_factor = mutation_scale_factor
        self.crossover_rate = crossover_rate
        self.early_stopping_after_gens = early_stopping_after_gens
        self.elite_size = elite_size
        self.sigma = sigma
        self.introduce_new_values_probability = introduce_new_values_probability

        if any(ele is None for ele in domain_size) and (density == None or number_particles == None):
            raise Exception("If you do not suppy a domain_size, you need to provide both the density and the number of particles.")
        elif density == None and number_particles == None:
            raise Exception("Please supply either the density or the number of particles.")
        elif density != None and not any(ele is None for ele in domain_size):
            self.density = density
            self.domain_size = domain_size
            self.number_particles = sprep.get_number_of_particles_for_constant_density(density, self.domain_size)
        elif number_particles and not any(ele is None for ele in domain_size):
            self.number_particles = number_particles
            self.domain_size = domain_size
            self.density = sprep.get_density(self.domain_size, self.number_particles)
        else:
            self.density = density
            self.number_particles = number_particles
            self.domain_size = sprep.get_domain_size_for_constant_density(self.density, self.number_particles)

        self.c_value_size = (self.number_particles-1) * 3
        if self.add_own_orientation:
            self.c_value_size += 1
        if self.add_random:
            self.c_value_size += 1

        self.selection_probabilities = shelp.normalise([1/(i+1) for i in range(self.population_size)], norm='l1')

        print(f"dom={self.domain_size}, d={self.density}, n={self.number_particles}")

    def __create_initial_population(self):
        rand_pop = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=((self.population_size, self.c_value_size)))
        if self.zero_choice_probability_initial != None:
            rand_pop[np.random.rand(*rand_pop.shape) < self.zero_choice_probability_initial] = 0
        return rand_pop
    
    def __get_orientation_difference_threshold_contribution(self, orientations):
        if self.orientations_difference_threshold == 2*np.pi:
            return 0 # in this case, they are allowed to turn as much as they like
        diff_gt_threshold = []
        for t in range(1, len(orientations)):
           orientations_t = sorient.compute_angles_for_orientations(orientations[t])
           orientations_t_minus_1 = sorient.compute_angles_for_orientations(orientations[t-1]) 
           diffs = np.absolute(orientations_t - orientations_t_minus_1)
           mask = diffs < self.orientations_difference_threshold
           masked_diffs = np.ma.MaskedArray(diffs, mask=mask, fill_value=0)
           diff_gt_threshold.extend(masked_diffs.compressed())
        percentage_flipflop = len(diff_gt_threshold)/(len(orientations)* len(orientations[0]))
        if percentage_flipflop > 0.7:
            return percentage_flipflop * 10
        return 0

    def __fitness_function(self, c_values):
        results = {t: [] for t in range(self.tmax)}
        c_values = self.__update_c_values(c_values)
        for i in range(self.num_iterations_per_individual):
            if self.start_order == 1 or (self.start_order == None and i < (self.num_iterations_per_individual/2)):
                initialState = sprep.create_ordered_initial_distribution_equidistanced_individual(domain_size=self.domain_size, number_particles=self.number_particles)
            else:
                initialState = (None, None, None)
            simulator = RunModel(domain_size=self.domain_size,
                                radius=self.radius,
                                noise=self.noise,
                                speed=self.speed,
                                number_particles=self.number_particles,
                                c_values=c_values,
                                add_own_orientation=self.add_own_orientation,
                                add_random=self.add_random)
            simulation_data = simulator.simulate(tmax=self.tmax, initialState=initialState)
            _, _, orientations = simulation_data
            [results[t].append(sorient.compute_global_order(orientations[t])) for t in range(self.tmax)]
        resultsArr = [np.average(results[t]) for t in range(self.tmax)]
        target = (self.changeover_point_timestep) * [self.start_order] + (self.tmax-self.changeover_point_timestep) * [self.target_order]
        resultsIntegral = integrate.simpson(y=resultsArr[self.start_timestep_evaluation: self.tmax], x=range(self.start_timestep_evaluation, self.tmax))
        targetIntegral = integrate.simpson(y=target[self.start_timestep_evaluation: self.tmax], x=range(self.start_timestep_evaluation, self.tmax))
        
        fitness = np.absolute(targetIntegral-resultsIntegral) / self.tmax

        return fitness, fitness + (self.c_values_norm_factor * shelp.normalise(values=c_values, norm='l0')) + self.__get_orientation_difference_threshold_contribution(orientations=orientations)        

    def __mutation(self, x):
        noise = np.random.normal(0, self.sigma, x.shape) 
        mutated_with_noise = x + noise

        rands = np.random.uniform(0, 1, size=x.shape)
        prob_noise = 1-self.introduce_new_values_probability-self.introduce_new_values_probability
        if self.introduce_new_values_probability > 0:
            new_vals = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=x.shape)
            mutated_with_noise = np.where(((rands > prob_noise) & (rands <= (1-self.zero_choice_probability_mutation))), new_vals, mutated_with_noise)
        if self.zero_choice_probability_mutation > 0:
            mutated_with_noise = np.where((rands > (1-self.zero_choice_probability_mutation)), 0, mutated_with_noise)        
        return mutated_with_noise
    
    def __check_bounds(self, mutated, bounds):
        mutated_bound = np.clip(mutated, bounds[0], bounds[1])
        return mutated_bound
    
    def __crossover(self, p1, p2):
        pt = random.randint(1, len(p1)-2)
        a = list(p1[:pt])
        b = list(p2[pt:])
        a.extend(b)
        return np.array(a)
    
    def __update_c_values(self, c_values):
        c_values = np.where(((c_values >= self.update_to_zero_bounds[0]) & (c_values <= self.update_to_zero_bounds[1])), 0, c_values)
        if self.use_norm == True:
            c_values = shelp.normalise(c_values, norm='l1')
        return c_values
    
    def __plot_fitnesses(self, fitnesses, save_path_plots=None):
        plt.plot(fitnesses)
        if save_path_plots:
            plt.savefig(f"{save_path_plots}.svg")
            plt.savefig(f"{save_path_plots}.jpeg")
        else:
            plt.show()         

    def run(self, save_path_plots=None, save_path_log=None, log_depth='all'):
        with open(f"{save_path_log}.csv", 'a', newline='') as log:
            w = csv.writer(log)
            headers = slog.create_headers(self.c_value_size)
            w.writerow(headers)
            log.flush()

            rng = np.random.default_rng()

            population  = self.__create_initial_population()

            last_improvement_at_gen = 0
            best_fitnesses_for_generations = []
            prev_fitness = None
            for generation in range(self.num_generations):
                print(f"gen {generation+1}/{self.num_generations}")
                fitnesses_both = np.array([self.__fitness_function(individual) for individual in population])

                fitnesses_order = fitnesses_both[:, 0]
                fitnesses = fitnesses_both[:, 1]

                sorted_indices = np.argsort(fitnesses)
                best_individual_index = sorted_indices[0]
                best_individual = population[best_individual_index]
                best_fitness = fitnesses[best_individual_index] 

                if prev_fitness == None or best_fitness < prev_fitness:
                    best_individual = population[np.argmin(fitnesses)]
                    prev_fitness = best_fitness
                    last_improvement_at_gen = generation
                    print('Iteration: %d f([%s]) = %.5f' % (generation, np.around(best_individual, decimals=5), best_fitness))

                                # saving the fitnesses
                if log_depth == 'all':
                    log_dict_list = slog.create_dicts_for_logging(generation, population, fitnesses, fitnesses_order)
                else:
                    log_dict_list = slog.create_dicts_for_logging(generation, [best_individual], [best_fitness], [fitnesses_order[best_individual_index]])
                for dict in log_dict_list:
                    w.writerow(dict.values())
                log.flush()

                best_fitnesses_for_generations.append(best_fitness)
                if self.early_stopping_after_gens != None and generation-last_improvement_at_gen > self.early_stopping_after_gens:
                    print(f"Early stopping at iteration {generation} after {self.early_stopping_after_gens} generations without improvement")
                    break

                # Creating the new population
                new_population = list(population[sorted_indices[:self.elite_size]])
                p1 = rng.choice(a=population, size=(self.population_size-self.elite_size), p=self.selection_probabilities, axis=0)
                # TODO prohibit choosing the same parent twice
                p2 = rng.choice(a=population, size=(self.population_size-self.elite_size), axis=0)
                for j in range(len(p1)):
                    child = self.__crossover(p1=p1[j], p2=p2[j])
                    child = self.__mutation(child)
                    child = self.__check_bounds(child, bounds=self.bounds)
                    new_population.append(child)

                population = np.array(new_population)

            self.__plot_fitnesses(best_fitnesses_for_generations, save_path_plots)
            return [best_individual, best_fitness, fitnesses_order[best_individual_index]]
