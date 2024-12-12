import numpy as np

from model.active_elastic.genetic_algorithm_model_active_elastic_basic import GeneticAlgorithmActiveElasticBasicWithParams
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
population_size = 30
num_gens = 20
num_iters = 30
num_iters_per_ind = 10
early_stopping = None

radius = 10
noise_percentage = 1
density = 0.01
tmax = 1000


bounds = [-5, 5]

num_c_values = n + 4

if target_order == 1:
    postfix = f"_test_ga_ae_basic_with_params_order_n={n}_pi={prob_init}_pm={prob_mut}_g={num_gens}_pop={population_size}"
elif target_order == 0:
    postfix = f"_test_ga_ae_basic_with_params_disorder_n={n}_pi={prob_init}_pm={prob_mut}_g={num_gens}_pop={population_size}"
else:
    postfix = f"_test_ga_ae_basic_with_params_middle_n={n}_pi={prob_init}_pm={prob_mut}_g={num_gens}_pop={population_size}"

print(postfix)

save_path_best = f"best{postfix}.csv"
save_path_best_normalised = f"best{postfix}_normalised.csv"
save_path_general = f"all{postfix}"
save_path_plot = f"plot{postfix}"

slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True, n=n, has_own=False, has_random=False, ranking_by=[False, False, False]), save_path=save_path_best)
slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True, n=n, has_own=False, has_random=False, ranking_by=[False, False, False]), save_path=save_path_best_normalised)

for i in range(num_iters):
    model = GeneticAlgorithmActiveElasticBasicWithParams(radius=radius, 
                        tmax=tmax, 
                        density=density, 
                        number_particles=n,
                        noise_percentage=noise_percentage,
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


    slog.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1], 'fitness_order': best[2]}], ranking_by=[False, False, False], prepare=True, save_path=save_path_best, n=n)
    slog.log_results_to_csv([{'iter': i, 'individual': shelp.normalise(np.array(best[0])), 'fitness': best[1], 'fitness_order': best[2]}], ranking_by=[False, False, False], prepare=True, save_path=save_path_best_normalised, n=n)





""" import numpy as np

from model.run_model_active_elastic_basic import ActiveElasticRunModel
import services.service_preparation as sprep
import services.service_logging as slog
import services.service_helper as shelp

from animator.animator_2d import Animator2D
from animator.animator_matplotlib import MatplotlibAnimator
from evaluator.evaluator_multi_comp import EvaluatorMultiAvgComp

domain_size = (31, 31)
n = 100
radius = 10
speed = 0.1
alpha = 0.5
beta = 0.5
spring_constant = 0.5
natural_distance = 10
sensing_noise = sprep.get_noise_amplitude_value_for_percentage(1)
actuation_noise = sprep.get_noise_amplitude_value_for_percentage(0.5)

c_values = np.ones(n)

model_params = {'number_particles': n, 'radius': radius, 'domain_size': domain_size}
tmax = 1000
save_path = "ae_basic_test"

simulator = ActiveElasticRunModel(domain_size=domain_size, 
                                  number_particles=n, 
                                  radius=radius, 
                                  c_values=c_values,
                                  speed=speed,
                                  alpha=alpha,
                                  beta=beta,
                                  spring_constant=spring_constant,
                                  natural_distance=natural_distance,
                                  sensing_noise=sensing_noise,
                                  actuation_noise=actuation_noise)

simulation_data = simulator.simulate(tmax=tmax)
times, positions, orientations = simulation_data

animator = MatplotlibAnimator(simulation_data, (domain_size[0], domain_size[1], 100), red_indices=[0])

# prepare the animator
#preparedAnimator = animator.prepare(Animator2D(modelParams), frames=modelParams["tmax"])
preparedAnimator = animator.prepare(Animator2D(model_params), frames=tmax)

preparedAnimator.save_animation(f"{save_path}.mp4")

 """

