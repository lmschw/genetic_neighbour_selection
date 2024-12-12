
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
    if len(orientations) == 0:
        return orientations

    zero_mask = orientations==[0,0]
    zero_masked = np.ma.MaskedArray(orientations, mask=zero_mask)
    normalised = orientations/(np.sqrt(np.sum(zero_masked**2,axis=1))[:,np.newaxis])
    return normalised.data

def normalize_angles(angles):
    if len(angles) == 0:
        return angles
    return (angles + (2*np.pi)) % (2*np.pi)

def calculate_mean_orientations(orientations):
    """
    Computes the average of the orientations for every individual.

    Params:
        - orientations (array of floats): the orientation of every individual at the current timestep

    Returns:
        An array of floats containing the new, normalised orientations of every individual
    """
    if len(orientations) == 0:
        return orientations
    
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
    if len(array) == 0:
        return array

    rij=array[:,np.newaxis,:]-array   
    rij = rij - domain_size*np.rint(rij/domain_size) #minimum image convention
    return np.sum(rij**2,axis=2)


def get_angle_differences(angles, return_absolute=False):
    """
    Computes the differences between all individuals for the values provided by the array.

    Params:
        - array (array of floats): the values to be compared

    Returns:
        An array of arrays of floats containing the difference between each pair of values.
    """
    if len(angles) == 0:
        return angles

    angles_positive = angles + np.pi
    diff = angles_positive[:,np.newaxis]-angles_positive   
    if return_absolute:
        return np.absolute(diff)
    return diff
    

def get_differences_sqrt(array, domain_size):
    return np.sqrt(get_differences(array, domain_size))

def remove_diagonal(matrix):
    """
    Removes the diagnoal from a matrix.

    Params:
        - matrix (np.array): the matrix from which the diagonal should be removed

    Returns:
        The matrix without the diagonal.
    """
    return matrix[~np.eye(len(matrix), dtype=bool)].reshape(len(matrix), -1)

def compute_global_order(orientations):
    """
    Computes the global order within the domain.

    Params:
        - orientations (np.array): the orientation of every particle in the domain

    Returns:
        The global order, a value in the range [0,1].
    """
    sumOrientation = np.sum(orientations[np.newaxis,:,:],axis=1)
    return np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), len(orientations))[0]

def compute_angles_for_orientations(orientations):
    """
    Computes the angle in radians based on the (u,v)-coordinates of the current orientation.

    Params:
        - orientation (array of floats): the current orientation in (u,v)-coordinates

    Returns:
        A float representin the angle in radians.
    """
    if orientations.ndim == 3:
        return np.arctan2(orientations[:, :, 1], orientations[:, :, 0])
    return np.arctan2(orientations[:, 1], orientations[:, 0])


def compute_uv_coordinates(angle):
    """
    Computes the (u,v)-coordinates based on the angle.

    Params:
        - angle (float): the angle in radians

    Returns:
        An array containing the [u, v]-coordinates corresponding to the angle.
    """
    # compute the uv-coordinates
    U = np.cos(angle)
    V = np.sin(angle)
    
    return [U,V]

def compute_uv_coordinates_for_list(angles):
    """
    Computes the (u,v)-coordinates based on the angle.

    Params:
        - angle (float): the angle in radians

    Returns:
        An array containing the [u, v]-coordinates corresponding to the angle.
    """
    # compute the uv-coordinates
    U = np.cos(angles)
    V = np.sin(angles)
    
    return np.column_stack((U,V))

def get_unit_vectors_for_angles(angles):
    return np.hstack((np.cos(angles)[:,None],np.sin(angles)[:,None]))

def get_perpendicular_vectors(vectors):
    new_vectors = []
    for v in vectors:
        if v[0] != 0:
            new_vectors.append([-(v[1]/v[0]), 1]) 
        elif v[1] != 0:
            new_vectors.append([1, -(v[0]/v[1])])
        else:
            new_vectors.append([1,1])
    return np.array(new_vectors)