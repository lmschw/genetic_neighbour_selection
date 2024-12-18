from sklearn import preprocessing
import numpy as np

import services.service_orientations as sorient

"""
Contains helper methods that do not fit the themes of the other services.
"""

def normalise(values, norm='l1', axis=1):
    """
    Applies the Euclidean norm to the values.

    Params:
        - values (np.array): the values to be normalised

    Returns:
        A numpy array with the normalised values.
    """
    if norm == 'l0':
        if values.ndim == 2:
            return np.count_nonzero(values, axis=1)
        return np.count_nonzero(values)
    
    is_one_dim = False
    if len(values) == 0:
        return values
    
    if values.ndim == 1:
        is_one_dim = True
        values = [values]

    normalised_vector = preprocessing.normalize(X=values, norm=norm, axis=axis)
    if is_one_dim:
        return normalised_vector[0]
    return normalised_vector

def get_neighbours(positions, domain_size, radius):
    """
    Determines all the neighbours for each individual.

    Params:
        - positions (array of floats): the position of every individual at the current timestep

    Returns:
        An array of arrays of booleans representing whether or not any two individuals are neighbours
    """
    rij2 = sorient.get_differences(positions, domain_size)
    return (rij2 <= radius**2)