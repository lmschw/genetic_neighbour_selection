import numpy as np

from model.genetic_algorithm_model_active_elastic import GeneticAlgorithmActiveElastic
import services.service_preparation as sprep
import services.service_logging as slog
import services.service_helper as shelp

n = 10
alpha = 0.005
prob_init = 0.9
prob_mut = 0.1
prob_intro = 0.05
start_order = 1
target_order = 0
population_size = 10
num_gens = 20
num_iters = 1
num_iters_per_ind = 1
early_stopping = None

add_ranking_by = [True, True, True]
add_own = True
add_random = False
bounds = [0.00001, 5]

num_c_values = 7
if add_own:
    num_c_values += 1
if add_random:
    num_c_values += 1

if target_order == 1:
    postfix = f"_test_ga_dist_order_n={n}_pi={prob_init}_pm={prob_mut}_g={num_gens}_pop={population_size}"
else:
    postfix = f"_test_ga_dist_disorder_n={n}_pi={prob_init}_pm={prob_mut}_g={num_gens}_pop={population_size}"

postfix = "_test_log"

print(postfix)

save_path_best = f"best{postfix}.csv"
save_path_best_normalised = f"best{postfix}_normalised.csv"
save_path_general = f"all{postfix}"
save_path_plot = f"plot{postfix}"

slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True, n=n, has_own=add_own, has_random=add_random, ranking_by=add_ranking_by), save_path=save_path_best)
slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True, n=n, has_own=add_own, has_random=add_random, ranking_by=add_ranking_by), save_path=save_path_best_normalised)

for i in range(num_iters):
    model = GeneticAlgorithmActiveElastic(radius=100, 
                        tmax=1000, 
                        density=0.01, 
                        number_particles=n,
                        noise_percentage=1,
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
                        target_order=target_order)

    best = model.run(save_path_log=save_path_general, save_path_plots=save_path_plot)
    print(f"BEST overall: {best}")


    slog.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1], 'fitness_order': best[2]}], ranking_by=add_ranking_by, prepare=True, save_path=save_path_best, n=n)
    slog.log_results_to_csv([{'iter': i, 'individual': shelp.normalise(np.array(best[0])), 'fitness': best[1], 'fitness_order': best[2]}], ranking_by=add_ranking_by, prepare=True, save_path=save_path_best_normalised, n=n)

