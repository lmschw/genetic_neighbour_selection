import numpy as np

from model.run_model import RunModel
from model.genetic_algorithm_model import GeneticAlgorithm
from event.event import ExternalStimulusOrientationChangeEvent
from event.enums_event import EventEffect, DistributionType, EventSelectionType
import services.service_preparation as sprep
import services.service_logging as slog
import services.service_helper as shelp

n = 10
alpha = 0.005
prob_init = 0.9
prob_mut = 0.1
prob_intro = 0.05
population_size = 30
num_gens = 20
num_iters = 25
num_iters_per_ind = 10
early_stopping = None
radius = 100
density = 0.01
domain_size = sprep.get_domain_size_for_constant_density(density=density, number_particles=n)

add_own = True
add_random = True
bounds = [-5, 5]

tmax = 3000
event_radius = radius
event_start = 1000
event_duration = 1000
event_effect = EventEffect.AWAY_FROM_ORIGIN
start_order = 1
target_order = 0
event_area = [[domain_size[0]/2, domain_size[1]/2, event_radius]]
event = ExternalStimulusOrientationChangeEvent(start_timestep=event_start,
                                               duration=event_duration,
                                               domain_size=domain_size,
                                               event_effect=event_effect,
                                               distribution_type=DistributionType.LOCAL_SINGLE_SITE,
                                               areas=event_area,
                                               radius=event_radius,
                                               number_of_affected=n/2,
                                               event_selection_type=EventSelectionType.NEAREST_DISTANCE,
                                               angle=np.pi
                                               )
events = [event]

num_c_values = 3 * (n-1)
if add_own:
    num_c_values += 1
if add_random:
    num_c_values += 1

postfix = f"_ga_event_{event_effect.val}_n={n}_pi={prob_init}_pm={prob_mut}_pn={prob_intro}_g={num_gens}_pop={population_size}"

print(postfix)

save_path_best = f"best{postfix}.csv"
save_path_best_normalised = f"best{postfix}_normalised.csv"
save_path_general = f"all{postfix}"
save_path_plot = f"plot{postfix}"

slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True), save_path=save_path_best)
slog.initialise_log_file_with_headers(slog.create_headers(num_c_values, is_best=True), save_path=save_path_best_normalised)

for i in range(num_iters):
    model = GeneticAlgorithm(radius=radius, 
                        tmax=tmax, 
                        density=density, 
                        number_particles=n,
                        noise_percentage=0,
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
                        target_order=target_order,
                        start_order=start_order,
                        changeover_point_timestep=event_start,
                        events=events)

    best = model.run(save_path_log=save_path_general, save_path_plots=save_path_plot)
    print(f"BEST overall: {best}")


    slog.log_results_to_csv([{'iter': i, 'individual': np.array(best[0]), 'fitness': best[1], 'fitness_order': best[2]}], prepare=True, save_path=save_path_best)
    slog.log_results_to_csv([{'iter': i, 'individual': shelp.normalise(np.array(best[0])), 'fitness': best[1], 'fitness_order': best[2]}], prepare=True, save_path=save_path_best_normalised)


