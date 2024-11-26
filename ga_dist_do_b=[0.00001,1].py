import numpy as np

from model.run_model import RunModel
from model.genetic_algorithm_model_distribution import GeneticAlgorithmDistribution
import services.service_preparation as sprep
import services.service_logging as slog
import services.service_helper as shelp

num_c_values = 7

n = 10
alpha = 0.005
prob_init = 0
prob_mut = 0
prob_intro = 0
start_order = 1
target_order = 0
population_size = 30
num_gens = 20
num_iters = 50
num_iters_per_ind = 10
early_stopping = None

add_own = True
add_random = False
bounds = [0.00001, 1]

tmax = 2000
radius = 100
noise_percentage = 1
speed = 0.5
density = 0.01

if target_order == 1:
    postfix = f"_test_ga_dist_order_n={n}_bounds={bounds}_g={num_gens}_pop={population_size}_noise={noise_percentage}_speed={speed}"
elif target_order == 0:
    postfix = f"_test_ga_dist_disorder_n={n}_bounds={bounds}_g={num_gens}_pop={population_size}_noise={noise_percentage}_speed={speed}"
else:
    postfix = f"_test_ga_dist_middle_n={n}_bounds={bounds}_g={num_gens}_pop={population_size}_noise={noise_percentage}_speed={speed}"

print(postfix)

save_path_best = f"best{postfix}.csv"
save_path_best_normalised = f"best{postfix}_normalised.csv"
save_path_general = f"all{postfix}"
save_path_plot = f"plot{postfix}"

slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True), save_path=save_path_best)
slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True), save_path=save_path_best_normalised)

for i in range(num_iters):
    model = GeneticAlgorithmDistribution(
                        radius=radius, 
                        tmax=tmax, 
                        density=density, 
                        number_particles=n,
                        noise_percentage=noise_percentage,
                        speed=speed,
                        bounds=bounds,
                        add_own_orientation=add_own,
                        add_random=add_random, 
                        c_values_norm_factor=alpha,
                        zero_choice_probability_initial=prob_init,
                        zero_choice_probability_mutation=prob_mut,
                        introduce_new_values_probability=prob_intro,
                        num_generations=num_gens, 
                        num_iterations_per_individual=num_iters_per_ind,
                        population_size=population_size,
                        early_stopping_after_gens=early_stopping,
                        target_order=target_order
                        )

    best = model.run(save_path_log=save_path_general, save_path_plots=save_path_plot)
    print(f"BEST overall: {best}")


    slog.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1], 'fitness_order': best[2]}], prepare=True, save_path=save_path_best)
    slog.log_results_to_csv([{'iter': i, 'individual': shelp.normalise(np.array(best[0])), 'fitness': best[1], 'fitness_order': best[2]}], prepare=True, save_path=save_path_best_normalised)


