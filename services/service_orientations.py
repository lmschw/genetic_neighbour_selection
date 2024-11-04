
import numpy as np

"""
Service that includes methods that manipulate orientations.
"""

def normalize_orientations(orientations):
    """
    Normalises the orientations of all particles for the current time step

    Parameters:
        - orientations (array): The current orientations of all particles

    Returns:
        The normalised orientations of all particles as an array.
    """
    return orientations/(np.sqrt(np.sum(orientations**2,axis=1))[:,np.newaxis])

def calculate_mean_orientations(orientations):
    """
    Computes the average of the orientations for every individual.

    Params:
        - orientations (array of floats): the orientation of every individual at the current timestep

    Returns:
        An array of floats containing the new, normalised orientations of every individual
    """
    mean_orientations = np.average(orientations, axis=1)
    return normalize_orientations(mean_orientations)

def get_differences(array, domain_size):
    """
    Computes the differences between all individuals for the values provided by the array.

    Params:
        - array (array of floats): the values to be compared

    Returns:
        An array of arrays of floats containing the difference between each pair of values.
    """
    rij=array[:,np.newaxis,:]-array   
    rij = rij - domain_size*np.rint(rij/domain_size) #minimum image convention
    return np.sum(rij**2,axis=2)

def remove_diagonal(array):
    return array[~np.eye(len(array), dtype=bool)].reshape(len(array), -1)

def compute_global_order(orientations):
    sumOrientation = np.sum(orientations[np.newaxis,:,:],axis=1)
    return np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), len(orientations))[0]
