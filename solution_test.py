import numpy as np

from animator.animator_2d import Animator2D
from animator.animator_matplotlib import MatplotlibAnimator
from evaluator.evaluator_multi_comp import EvaluatorMultiAvgComp
from model.run_model import RunModel
import services.service_saved_run_model as ssave
import services.service_preparation as sprep

best_individual_order = np.array([-1.        , -1.        , -0.953106  ,  0.68805811, -0.7195935 ,
        0.96470597, -1.        , -1.        , -0.62461648, -0.31312997,
       -1.        , -1.        , -0.89212112, -1.        , -1.        ,
       -1.        , -0.66975609, -0.8939383 ,  0.00859001, -0.74237459,
       -1.        , -1.        , -1.        , -1.        , -0.65892004,
        0.00451248,  0.4406018 ])

best_individual_disorder = np.array([-0.85863, -0.68543,  0.4232 , -0.08102 , 0.59685 , 0.07711 , 1.    ,  -0.00908,
 -0.00274, -0.17459 , 0.22859 ,-0.0697,   0.00793 ,-0.7475  , 0.56477 , 0.06464,
  0.83335,  0.90827 ,-0.37688 ,-0.10139 , 0.61347 ,-0.37269 , 1.   ,    1.,
  0.19797, -1.  ,     0.34839])

farthest = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
nearest = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
hod = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
lod = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
add_own_orientations = True # add the particle's own orientation at the end

farthest = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
add_own_orientations = False

test_2 = np.array([1,1,-1,-0.0391887,-1,-0.04812661,1,0.60729156,-0.78049631,-1,-0.87177953,0.80228549,-0.72711081,1,1,-0.40070393,-1,-0.60916819,0.7687721,0.15342928,-0.19759769,-0.22476174,0.03415772,0.53463007,0.47143451,0.7476541,-0.20561136])
test_1 = np.array([0.64490686, -0.36929425, 1,         -0.1873528,   0.70981314, -0.62358704,  1,         -0.80142409,  0.77438957, -1,          0.8370958,   0.00381771,  1,         -0.40013243, -0.83195211,  0.97873915, -0.10991523,  0.81707503, -0.71840685, -0.35307178, -0.62066749, -0.27144979, -1,         -0.23242119,  0.06301562,  1,         -0.03621834])
test_3_half = np.array([-0.161,   -0.65302, -0.21428, -1,       1,      -0.70602,  0.69272, -0.36237, -0.88507, -0.07826, -0.53279,  1,       0.61919, -0.17625,  0.82348,  0.26612,  0.37776, -0.36539,  0.13395,  0.57048, -1,       0.75766,  0.45333, -0.84249, -1,      -0.31863, -1])
test_3 = np.array([-1,-1,-1,-1,0.49917502,-0.18865848,-1,-0.65079671,1,-0.09974339,0.22934156,0.6527323,1,-1,-1,0.30156005,1,0.58031856,0.09244338,-0.02481739,1,0.15187964,0.51974084,-0.44777745,0.54964162,-0.68214867,0.04631725])
test_4 = np.array([1.0,-0.7156773689847655,-0.7360633216339458,0.35279285470486466,0.9930476153362653,0.5999952359792813,-0.8446329951592972,1.0,0.741333905050801,0.9217079428505963,-0.17730664350330216,-0.05023843836313446,0.9505837388214176,0.45468305261738484,1.0,-0.14510749282618995,0.03329256570921513,1.0,-0.4622261675876065,1.0,-0.5580013370887591,-0.6010484555496782,-1.0,0.3690755983186329,-0.4359787580911174,0.9415159697510929,0.6937667154284237])

best_individual = test_4

base_filename = "test_4"

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
                    c_values=best_individual,
                    add_own_orientation=add_own_orientations)

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


