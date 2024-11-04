import numpy as np

def get_noise_amplitude_value_for_percentage(percentage):
    """
    Paramters:
        - percentage (int, 1-100)
    """
    return 2 * np.pi * (percentage/100)