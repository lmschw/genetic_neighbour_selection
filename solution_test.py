import numpy as np

from animator.animator_2d import Animator2D
from animator.animator_matplotlib import MatplotlibAnimator
from evaluator.evaluator_multi_comp import EvaluatorMultiAvgComp
from model.run_model import RunModel
import services.service_saved_run_model as ssave
import services.service_saved_c_values as ssavedc
import services.service_preparation as sprep

add_own_orientations = True
add_random = False

test_norm = np.array([-0.013081037,	-0.036045735,	-0.000323752,	0.022198250817864976,	-0.00303675,	-0.050044312,	-0.03233667,	0.009414294,	-0.042298389,	0.046853972,	0.068945726,	-0.026213319,	-0.000147059,	0.027333175771849814,	-0.017342028,	0.032234123881716696,	-0.103540163,	-0.063463532,	-0.006785963,	-0.036194692,	0.030902432110207337,	0,	0.1673728559312047,	0.085840216,	-0.02039786,	0,	0.057653692,	0

])

best_individual = test_norm

base_filename = "test_middle_limited_6"
location = "c:/Users/lschw/Downloads/results_lower_speed/"
base_filename = "results_lower_speed/best_test_ga_disorder_n=10_pi=0.9_pm=0.1_pn=0.05_g=20_pop=30_noise=1_speed=0.1_normalised"
test_norm = ssavedc.load_solution_as_nparray(iter=100, filepath=f"{location}{base_filename}.csv")


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
    simulator = RunModel(domain_size=domain_size,
                    radius=radius,
                    noise=noise,
                    speed=speed,
                    number_particles=n,
                    c_values=best_individual,
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


