import numpy as np

from model.run_model import RunModel
from model.differential_evolution_model import DifferentialEvolution
import services.service_preparation as sprep
import services.service_logging as slog
import services.service_helper as shelp

n = 10
target_order = 0
population_size = 30
num_gens = 20
num_iters = 5

add_own = True
add_random = True
bounds = [-5, 5]
zero_bounds = [-0.5, 0.5]

num_c_values = 3 * (n-1)
if add_own:
    num_c_values += 1
if add_random:
    num_c_values += 1

for a in [0.01, 0.05, 0.1]:
    postfix = f"_own_random_disorder_zeros_n={n}_b5_a={a}"

    print(postfix)

    save_path_best = f"best{postfix}.csv"
    save_path_best_normalised = f"best{postfix}_normalised.csv"
    save_path_general = f"all{postfix}"
    save_path_plot = f"plot{postfix}"

    slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True), save_path=save_path_best)
    slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True), save_path=save_path_best_normalised)

    for i in range(num_iters):
        model = DifferentialEvolution(radius=100, 
                            tmax=1000, 
                            density=0.01, 
                            number_particles=n,
                            noise_percentage=0,
                            add_own_orientation=add_own,
                            add_random=add_random, 
                            c_values_norm_factor=a,
                            num_generations=num_gens, 
                            num_iterations_per_individual=10,
                            population_size=population_size,
                            bounds=bounds,
                            update_to_zero_bounds=zero_bounds,
                            early_stopping_after_gens=None,
                            target_order=target_order)

        best = model.run(save_path_log=save_path_general, save_path_plots=save_path_plot)
        print(f"BEST overall: {best}")


        slog.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1]}], prepare=True, save_path=save_path_best)
        slog.log_results_to_csv([{'iter': i, 'individual': shelp.normalise(np.array(best[0])), 'fitness': best[1]}], prepare=True, save_path=save_path_best_normalised)
