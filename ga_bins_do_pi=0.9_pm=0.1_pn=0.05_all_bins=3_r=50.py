import numpy as np

from model.genetic_algorithm_model_bins import GeneticAlgorithmDistribution
import services.service_preparation as sprep
import services.service_logging as slog
import services.service_helper as shelp

n = 10
radius = 50
alpha = 0.005
prob_init = 0.9
prob_mut = 0.3
prob_intro = 0.05
start_order = 1
target_order = 0
population_size = 30
num_gens = 20
num_iters = 30
num_iters_per_ind = 10
early_stopping = None

add_ranking_by = [True, True, True]
add_own = False
add_random = False
bounds = [-5, 5]
#bin_bounds = [0, 0.33, 0.66, 1]
bin_bounds = [None]
num_bins = 3

tmax = 2000
speed = 0.5
noise_percentage = 1
density = 0.01

num_c_values = 27
if add_own:
    num_c_values += 1
if add_random:
    num_c_values += 1

if target_order == 1:
    postfix = f"_test_ga_bins_order_n={n}_bounds={bounds}_alpha={alpha}_bins={num_bins}_pinit={prob_init}_pmut={prob_mut}_pintro={prob_intro}_g={num_gens}_pop={population_size}_noise={noise_percentage}_speed={speed}_r={radius}"
elif target_order == 0:
    postfix = f"_test_ga_bins_disorder_n={n}_bounds={bounds}_alpha={alpha}_bins={num_bins}_pinit={prob_init}_pmut={prob_mut}_pintro={prob_intro}_g={num_gens}_pop={population_size}_noise={noise_percentage}_speed={speed}_r={radius}"
else:
    postfix = f"_test_ga_bins_middle_n={n}_bounds={bounds}_alpha={alpha}_bins={num_bins}_pinit={prob_init}_pmut={prob_mut}_pintro={prob_intro}_g={num_gens}_pop={population_size}_noise={noise_percentage}_speed={speed}_r={radius}"

print(postfix)

save_path_best = f"best{postfix}.csv"
save_path_best_normalised = f"best{postfix}_normalised.csv"
save_path_general = f"all{postfix}"
save_path_plot = f"plot{postfix}"

slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True, n=n, has_own=add_own, has_random=add_random, ranking_by=add_ranking_by), save_path=save_path_best)
slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True, n=n, has_own=add_own, has_random=add_random, ranking_by=add_ranking_by), save_path=save_path_best_normalised)

for i in range(num_iters):
    print(f"iter {i+1}/{num_iters}")
    model = GeneticAlgorithmDistribution(radius=radius, 
                        tmax=tmax, 
                        density=density, 
                        number_particles=n,
                        noise_percentage=noise_percentage,
                        speed=speed,
                        add_ranking_by=add_ranking_by,
                        add_own_orientation=add_own,
                        add_random=add_random, 
                        bounds=bounds,
                        c_values_norm_factor=alpha,
                        zero_choice_probability_initial=prob_init,
                        zero_choice_probability_mutation=prob_mut,
                        introduce_new_values_probability=prob_intro,
                        num_generations=num_gens, 
                        num_iterations_per_individual=num_iters_per_ind,
                        population_size=population_size,
                        early_stopping_after_gens=early_stopping,
                        start_order=start_order,
                        target_order=target_order,
                        bin_bounds=bin_bounds,
                        num_bins=num_bins)

    best = model.run(save_path_log=save_path_general, save_path_plots=save_path_plot)
    print(f"BEST overall: {best}")


    slog.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1], 'fitness_order': best[2]}], ranking_by=add_ranking_by, prepare=True, save_path=save_path_best, n=n)
    slog.log_results_to_csv([{'iter': i, 'individual': shelp.normalise(np.array(best[0])), 'fitness': best[1], 'fitness_order': best[2]}], ranking_by=add_ranking_by, prepare=True, save_path=save_path_best_normalised, n=n)


