import numpy as np
import scipy.integrate as integrate
import csv
from sklearn import preprocessing

from model.run_model import RunModel
import services.service_preparation as sprep
import services.service_orientations as sorient
import services.service_logging as slog


class GeneticModel:
    def __init__(self, radius, tmax, domain_size=(None, None), density=None, number_particles=None, speed=1, noise_percentage=1, 
                 num_generations=1000, num_iterations_per_individual=10, add_own_orientation=False, add_random=False, start_timestep_evaluation=0, 
                 changeover_point_timestep=0, start_order=0, target_order=1, population_size=100, bounds=[-1, 1], 
                 mutation_scale_factor=1, crossover_rate=0.5, early_stopping_after_gens=None):
        self.radius = radius
        self.num_generations = num_generations
        self.num_iterations_per_individual = num_iterations_per_individual
        self.speed = speed
        self.noise_percentage = noise_percentage
        self.noise = sprep.get_noise_amplitude_value_for_percentage(self.noise_percentage)

        self.tmax = tmax
        self.add_own_orientation = add_own_orientation
        self.add_random = add_random
        self.start_timestep_evaluation = start_timestep_evaluation
        self.changeover_point_timestep = changeover_point_timestep
        self.start_order = start_order
        self.target_order = target_order
        self.population_size = population_size
        self.bounds = bounds
        self.mutation_scale_factor = mutation_scale_factor
        self.crossover_rate = crossover_rate
        self.early_stopping_after_gens = early_stopping_after_gens

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

        print(f"dom={self.domain_size}, d={self.density}, n={self.number_particles}")

    def __normalise(self, c_values):
        normalised_vector = preprocessing.normalize(X=[c_values], norm='l2')[0]
        return c_values/normalised_vector


    def __create_initial_population(self):
        return np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=((self.population_size, self.c_value_size)))

    def __fitness_function(self, c_values):
        results = {t: [] for t in range(self.tmax)}
        for i in range(self.num_iterations_per_individual):
            if i < (self.num_iterations_per_individual/2):
                initialState = sprep.create_ordered_initial_distribution_equidistanced_individual(domain_size=self.domain_size, number_particles=self.number_particles)
            else:
                initialState = (None, None, None)
            simulator = RunModel(domain_size=self.domain_size,
                                radius=self.radius,
                                noise=self.noise,
                                speed=self.speed,
                                number_particles=self.number_particles,
                                c_values=self.__normalise(c_values),
                                add_own_orientation=self.add_own_orientation,
                                add_random=self.add_random)
            simulation_data = simulator.simulate(tmax=self.tmax, initialState=initialState)
            _, _, orientations = simulation_data
            [results[t].append(sorient.compute_global_order(orientations[t])) for t in range(self.tmax)]
        resultsArr = [np.average(results[t]) for t in range(self.tmax)]
        target = (self.changeover_point_timestep) * [self.start_order] + (self.tmax-self.changeover_point_timestep) * [self.target_order]
        resultsIntegral = integrate.simpson(y=resultsArr[self.start_timestep_evaluation: self.tmax], x=range(self.start_timestep_evaluation, self.tmax))
        targetIntegral = integrate.simpson(y=target[self.start_timestep_evaluation: self.tmax], x=range(self.start_timestep_evaluation, self.tmax))
        if self.target_order == 1:
            return targetIntegral-resultsIntegral
        return resultsIntegral-targetIntegral
    
    def __mutation(self, x, F):
        return x[0] + F * (x[1] - x[2])
    
    def __check_bounds(self, mutated, bounds):
        mutated_bound = np.clip(mutated, bounds[0], bounds[1])
        return mutated_bound
    
    def __crossover(self, mutated, target, cr):
        # generate a uniform random value for every dimension
        p = np.random.rand(self.c_value_size)
        # generate trial vector by binomial crossover
        trial = [mutated[i] if p[i] < cr else target[i] for i in range(self.c_value_size)]
        return np.array(trial)

    def run(self, save_path=None, log_depth='all'):
        with open(f"{save_path}.csv", 'a', newline='') as log:
            w = csv.writer(log)
            headers = slog.create_headers(self.number_particles)
            w.writerow(headers)
            log.flush()
            population  = self.__create_initial_population()
            fitnesses = [self.__fitness_function(individual) for individual in population]
            best_individual = population[np.argmin(fitnesses)]
            best_fitness = min(fitnesses)
            prev_fitness = best_fitness
            # saving the fitnesses
            if log_depth == 'all':
                log_dict_list = slog.create_dicts_for_logging(-1, population, fitnesses)
            else:
                log_dict_list = slog.create_dicts_for_logging(-1, [best_individual], [best_fitness])
            for dict in log_dict_list:
                w.writerow(dict.values())
            log.flush()
            last_improvement_at_gen = 0
            for iter in range(self.num_generations):
                print(f"gen {iter+1}/{self.num_generations}")
                for ind in range(self.population_size):
                    candidates = [candidate for candidate in range(self.population_size) if candidate != ind]
                    a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                    mutated = self.__mutation([a, b, c], self.mutation_scale_factor)
                    mutated = self.__check_bounds(mutated, self.bounds)
                    trial = self.__crossover(mutated, population[ind], self.crossover_rate)
                    target_existing = fitnesses[ind]
                    target_trial = self.__fitness_function(trial)
                    if target_trial < target_existing:
                        population[ind] = trial
                        fitnesses[ind] = target_trial
                best_fitness = min(fitnesses)
                if best_fitness < prev_fitness:
                    best_individual = population[np.argmin(fitnesses)]
                    prev_fitness = best_fitness
                    last_improvement_at_gen = iter
                    print('Iteration: %d f([%s]) = %.5f' % (iter, np.around(best_individual, decimals=5), best_fitness))
                # saving the fitnesses
                if log_depth == 'all':
                    log_dict_list = slog.create_dicts_for_logging(iter, population, fitnesses)
                else:
                    log_dict_list = slog.create_dicts_for_logging(iter, [best_individual], [best_fitness])
                for dict in log_dict_list:
                    w.writerow(dict.values())
                log.flush()
                if self.early_stopping_after_gens != None and iter-last_improvement_at_gen > self.early_stopping_after_gens:
                    print(f"Early stopping at iteration {iter} after {self.early_stopping_after_gens} generations without improvement")
                    break
            return [best_individual, best_fitness]
