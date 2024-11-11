import numpy as np

from model.run_model import RunModel
from model.genetic_model import GeneticModel
import services.service_preparation as sprep
import services.service_logging as slog
import services.service_helper as shelp

n = 10
postfix = f"_own_random_order_basic_n={n}_b5"

print(postfix)

save_path_best = f"best{postfix}.csv"
save_path_best_normalised = f"best{postfix}_normalised.csv"
save_path_general = f"all{postfix}"
save_path_plot = f"plot{postfix}"

add_own = True
add_random = True
bounds = [-5, 5]

num_c_values = 3 * (n-1)
if add_own:
    num_c_values += 1
if add_random:
    num_c_values += 1


slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True), save_path=save_path_best)
slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True), save_path=save_path_best_normalised)

for i in range(50):
    model = GeneticModel(radius=100, 
                        tmax=1000, 
                        density=0.01, 
                        number_particles=n,
                        noise_percentage=0,
                        add_own_orientation=add_own,
                        add_random=add_random, 
                        num_generations=20, 
                        num_iterations_per_individual=10,
                        population_size=20,
                        bounds=bounds,
                        early_stopping_after_gens=None,
                        target_order=1)

    best = model.run(save_path_log=save_path_general, save_path_plots=save_path_plot)
    print(f"BEST overall: {best}")


    slog.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1]}], prepare=True, save_path=save_path_best)
    slog.log_results_to_csv([{'iter': i, 'individual': shelp.normalise(np.array(best[0])), 'fitness': best[1]}], prepare=True, save_path=save_path_best_normalised)


