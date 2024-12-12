import numpy as np

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

