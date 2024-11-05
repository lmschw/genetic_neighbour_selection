import numpy as np

from animator.animator_2d import Animator2D
from animator.animator_matplotlib import MatplotlibAnimator
from evaluator.evaluator_multi_comp import EvaluatorMultiAvgComp
from model.run_model import RunModel
import services.service_saved_run_model as ssave
import services.service_preparation as sprep

best_individual = np.array([-0.85863, -0.68543,  0.4232 , -0.08102 , 0.59685 , 0.07711 , 1.    ,  -0.00908,
 -0.00274, -0.17459 , 0.22859 ,-0.0697,   0.00793 ,-0.7475  , 0.56477 , 0.06464,
  0.83335,  0.90827 ,-0.37688 ,-0.10139 , 0.61347 ,-0.37269 , 1.   ,    1.,
  0.19797, -1.  ,     0.34839])

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


