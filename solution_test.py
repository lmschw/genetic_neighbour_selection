import numpy as np

from animator.animator_2d import Animator2D
from animator.animator_matplotlib import MatplotlibAnimator
from evaluator.evaluator_multi_comp import EvaluatorMultiAvgComp
from model.run_model import RunModel
from model.run_model_active_elastic import ActiveElasticRunModel
from model.run_model_distribution import DistributionRunModel
import services.service_saved_run_model as ssave
import services.service_saved_c_values as ssavedc
import services.service_preparation as sprep

add_own_orientations = True
add_random = False

location = ""
base_filename = "best_test_log_normalised"
test_norm = ssavedc.load_solution_as_nparray(iter=0, filepath=f"{location}{base_filename}.csv")


tmax = 10000
n = 10
radius = 100
domain_size = (31.622776601683793, 31.622776601683793)
speed = 0.1
noise_percentage = 1
noise = sprep.get_noise_amplitude_value_for_percentage(noise_percentage)

model_params_arr = []
simulation_data_arr = []
for i in range(10):
    simulator = ActiveElasticRunModel(domain_size=domain_size,
                    radius=radius,
                    noise=noise,
                    speed=speed,
                    number_particles=n,
                    c_values=test_norm,
                    add_own_orientation=add_own_orientations,
                    add_random=add_random)

    simulation_data = simulator.simulate(tmax=tmax)

    model_params_arr.append(simulator.get_parameter_summary())
    simulation_data_arr.append(simulation_data)

    save_path = f"{base_filename}_{i}.json"
    ssave.save_model(simulation_data=simulation_data,
                    path=save_path,
                    model_params=simulator.get_parameter_summary())

# PLOT
labels = [""]
x_axis_label = "timesteps"
y_axis_label = "order"
evaluator = EvaluatorMultiAvgComp(model_params=[model_params_arr], simulation_data=[simulation_data_arr], evaluation_timestep_interval=1)
plot_save_path = f"order_{base_filename}.jpeg"
evaluator.evaluate_and_visualize(labels=labels, x_label=x_axis_label, y_label=y_axis_label, save_path=plot_save_path) 

# ANIMATION
model_params, simulation_data_loaded = ssave.load_model(save_path)

# Initalise the animator
animator = MatplotlibAnimator(simulation_data, (domain_size[0], domain_size[1], 100))

# prepare the animator
#preparedAnimator = animator.prepare(Animator2D(modelParams), frames=modelParams["tmax"])
preparedAnimator = animator.prepare(Animator2D(model_params), frames=tmax)

preparedAnimator.save_animation(f"{save_path}.mp4")


