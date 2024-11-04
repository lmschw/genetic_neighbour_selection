import numpy as np

from model.run_model import RunModel
import services.service_preparation as sprep


c_values = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]) # farthest

noise = sprep.get_noise_amplitude_value_for_percentage(1)
simulator = RunModel(domain_size=(10, 10), radius=20, noise=noise, speed=1, number_particles=4, c_values=c_values)
simulator.simulate(tmax=3000)
