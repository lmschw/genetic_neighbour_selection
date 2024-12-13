import numpy as np

from model.active_elastic.genetic_algorithm_model_active_elastic_chate import GeneticAlgorithmActiveElastic
from model.active_elastic.run_model_active_elastic_chate import ActiveElasticRunModel
import services.service_preparation as sprep
import services.service_logging as slog
import services.service_helper as shelp
import services.service_orientations as sorient

from animator.animator_2d import Animator2D
from animator.animator_matplotlib import MatplotlibAnimator
from evaluator.evaluator_multi_comp import EvaluatorMultiAvgComp

domain_size = (31, 31)
n = 10
radius = 10
speed = 0.1
mu = 0.1
kappa = 0.1
model_params = {'number_particles': n, 'radius': radius, 'domain_size': domain_size}
tmax = 1000
save_path = "ae_test"

simulator = ActiveElasticRunModel(domain_size=domain_size, 
                                  number_particles=n, 
                                  radius=radius, 
                                  speed=speed,
                                  mu=1.5, 
                                  kappa=0.1,
                                  noise=sprep.get_noise_amplitude_value_for_percentage(1),
                                  rotational_noise=1)

simulation_data = simulator.simulate(tmax=tmax)
times, positions, orientations = simulation_data

print(f"order={sorient.compute_global_order(orientations=orientations[-1])}")

animator = MatplotlibAnimator(simulation_data, (domain_size[0], domain_size[1], 100), red_indices=[0])

# prepare the animator
#preparedAnimator = animator.prepare(Animator2D(modelParams), frames=modelParams["tmax"])
preparedAnimator = animator.prepare(Animator2D(model_params), frames=tmax)

preparedAnimator.save_animation(f"{save_path}.mp4")

