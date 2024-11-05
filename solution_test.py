import numpy as np

from animator.animator_2d import Animator2D
from animator.animator_matplotlib import MatplotlibAnimator
from evaluator.evaluator_multi_comp import EvaluatorMultiAvgComp
from model.run_model import RunModel
import services.service_saved_run_model as ssave
import services.service_preparation as sprep

best_individual = np.array([-1.        , -1.        , -0.953106  ,  0.68805811, -0.7195935 ,
        0.96470597, -1.        , -1.        , -0.62461648, -0.31312997,
       -1.        , -1.        , -0.89212112, -1.        , -1.        ,
       -1.        , -0.66975609, -0.8939383 ,  0.00859001, -0.74237459,
       -1.        , -1.        , -1.        , -1.        , -0.65892004,
        0.00451248,  0.4406018 ])

tmax = 1000
n = 10
radius = 100
domain_size = (31.622776601683793, 31.622776601683793)
speed = 1
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
                    c_values=best_individual)

    simulation_data = simulator.simulate(tmax=tmax)

    model_params_arr.append(simulator.get_parameter_summary())
    simulation_data_arr.append(simulation_data)

    save_path = f"test_run_{i}.json"
    ssave.save_model(simulation_data=simulation_data,
                    path=save_path,
                    model_params=simulator.get_parameter_summary())

# PLOT
labels = [""]
x_axis_label = "timesteps"
y_axis_label = "order"
evaluator = EvaluatorMultiAvgComp(model_params=[model_params_arr], simulation_data=[simulation_data_arr], evaluation_timestep_interval=1)
plot_save_path = f"order_test_run.jpeg"
evaluator.evaluate_and_visualize(labels=labels, x_label=x_axis_label, y_label=y_axis_label, save_path=plot_save_path) 

# ANIMATION
model_params, simulation_data_loaded = ssave.load_model(save_path)

# Initalise the animator
animator = MatplotlibAnimator(simulation_data, (domain_size[0], domain_size[1], 100))

# prepare the animator
#preparedAnimator = animator.prepare(Animator2D(modelParams), frames=modelParams["tmax"])
preparedAnimator = animator.prepare(Animator2D(model_params), frames=tmax)

preparedAnimator.save_animation(f"{save_path}.mp4")


