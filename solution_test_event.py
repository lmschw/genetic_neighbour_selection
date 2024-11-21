import numpy as np

from animator.animator_2d import Animator2D
from animator.animator_matplotlib import MatplotlibAnimator
from evaluator.evaluator_multi_comp import EvaluatorMultiAvgComp
from model.run_model import RunModel
from event.event import ExternalStimulusOrientationChangeEvent
from event.enums_event import EventEffect, DistributionType, EventSelectionType
import services.service_saved_run_model as ssave
import services.service_preparation as sprep

add_own_orientations = True
add_random = True

test_norm = np.array([0.0,0.0,0.0,0.3081419208210972,0.0,0.0,0.0,0.21897607606158467,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.20978130968413883,0.0,0.0,0.0,0.0,0.0,0.2631006934331793

])

test_norm = np.array([0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.68688391,  0.69723981,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        , -0.45036562,
        0.        ,  0.        ,  0.        ,  0.    ])

best_individual = test_norm

base_filename = "test_event_distant_3"

n = 10
radius = 100
domain_size = (31.622776601683793, 31.622776601683793)
speed = 1
noise_percentage = 1
noise = sprep.get_noise_amplitude_value_for_percentage(noise_percentage)
noise = 0

tmax = 3000
event_radius = radius
event_start = 1000
event_duration = 1000
event_effect = EventEffect.ALIGN_TO_FIXED_ANGLE
start_order = 0
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
                    add_random=add_random,
                    events=events)

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


