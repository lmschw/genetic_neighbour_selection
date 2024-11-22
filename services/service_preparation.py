import numpy as np
import random

def get_alpha_for_n(n, ranking_by, has_own=False, has_random=False):
    size = (n-1) * np.count_nonzero(ranking_by)
    if has_own:
        size += 1
    if has_random:
        size += 1
    return 0.14 / size

def get_noise_amplitude_value_for_percentage(percentage):
    """
    Computes the noise amplitude for a given percentage.

    Params:
        - percentage (int, 1-100)

    Returns:
        The noise amplitude, a value in the range [0, 2pi]
    """
    return 2 * np.pi * (percentage/100)

def get_number_of_particles_for_constant_density(density, domain_size):
    """
    Computes the number of particles to keep the density constant for the supplied domain size.
    Density formula: "density" = "number of particles" / "domain area"

    Params:
        - density (float): the desired constant density of the domain
        - domain_size (tuple): tuple containing the x and y dimensions of the domain size

    Returns:
        The number of particles to be placed in the domain that corresponds to the density.
    """
    return int(density * (domain_size[0] * domain_size[1])) # density * area

def get_density(domain_size, number_of_particles):
    """
    Computes the density of a given system.
    Density formula: "density" = "number of particles" / "domain area"

    Params:
        - domain_size (tuple): tuple containing the x and y dimensions of the domain size
        - number_particles (int): the number of particles to be placed in the domain

    Returns:
        The density of the system as a float.
    """
    return number_of_particles / (domain_size[0] * domain_size[1]) # n / area


def get_domain_size_for_constant_density(density, number_particles):
    """
    Computes the domain size to keep the density constant for the supplied number of particles.
    Density formula: "density" = "number of particles" / "domain area"

    Params:
        - density (float): the desired constant density of the domain
        - number_particles (int): the number of particles to be placed in the domain

    Returns:
        A tuple containing the x and y dimensions of the domain size that corresponds to the density.
    """
    area = number_particles / density
    return (np.sqrt(area), np.sqrt(area))

def create_ordered_initial_distribution_equidistanced_individual(domain_size, number_particles, angle_x=None, angle_y=None):
    """
    Creates an ordered, equidistanced initial distribution of particles in a domain ready for use in individual decision scenarios. 
    The particles are placed in a grid-like fashion. The orientation of the particles is random unless specified
    but always the same for all particles.

    Params:
        - domain_size (tuple): tuple containing the x and y dimensions of the domain size
        - number_particles (int): the number of particles to be placed in the domain
        - angle_x (float [0,1)): first angle component to specify the orientation of all particles
        - angle_y (float [0,1)): second angle component to specify the orientation of all particles

    Returns:
        Positions and orientations for all particles within the domain. Can be used as the initial state of a Vicsek simulation.
    """
    positions, orientations = create_ordered_initial_distribution_equidistanced(domain_size, number_particles, angle_x, angle_y)
    return positions, orientations


def create_ordered_initial_distribution_equidistanced(domain_size, number_particles, angle_x=None, angle_y=None):
    """
    Creates an ordered, equidistanced initial distribution of particles in a domain. 
    The particles are placed in a grid-like fashion. The orientation of the particles is random unless specified
    but always the same for all particles.

    Params:
        - domain_size (tuple): tuple containing the x and y dimensions of the domain size
        - number_particles (int): the number of particles to be placed in the domain
        - angle_x (float [0,1)): first angle component to specify the orientation of all particles
        - angle_y (float [0,1)): second angle component to specify the orientation of all particles

    Returns:
        Positions and orientations for all particles within the domain. Can be used as the initial state of a Vicsek simulation.
    """
    # choose random angle for orientations
    if angle_x is None:
        angle_x = random.random()
    if angle_y is None:
        angle_y = random.random()

    # prepare the distribution for the positions
    x_length = domain_size[0]
    y_length = domain_size[1]
    
    area = x_length * y_length
    point_area = area / number_particles
    length = np.sqrt(point_area)

    # initialise the initialState components
    positions = np.zeros((number_particles, 2))
    orientations = np.zeros((number_particles, 2))

    # set the orientation for all particles
    orientations[:, 0] = angle_x
    orientations[:, 1] = angle_y

    counter = 0
    # set the position of every particle
    for x in np.arange(length/2, x_length, length):
        for y in np.arange(length/2, y_length, length):
            if counter < number_particles:
                positions[counter] = [x,y]
            counter += 1

    return positions, orientations


def create_ordered_initial_distribution_equidistanced_for_low_numbers(domain_size, number_particles, angle_x=None, angle_y=None):
    """
    Creates an ordered, equidistanced initial distribution of particles in a domain. 
    The particles are placed in a grid-like fashion. The orientation of the particles is random unless specified
    but always the same for all particles.

    Params:
        - domain_size (tuple): tuple containing the x and y dimensions of the domain size
        - number_particles (int): the number of particles to be placed in the domain
        - angle_x (float [0,1)): first angle component to specify the orientation of all particles
        - angle_y (float [0,1)): second angle component to specify the orientation of all particles

    Returns:
        Positions and orientations for all particles within the domain. Can be used as the initial state of a Vicsek simulation.

    """
    # choose random angle for orientations
    if angle_x is None:
        angle_x = random.random()
    if angle_y is None:
        angle_y = random.random()

    # prepare the distribution for the positions
    x_length = domain_size[0]
    y_length = domain_size[1]
    
    area = x_length * y_length
    point_area = area / number_particles
    length = np.sqrt(point_area)

    # initialise the initialState components
    positions = np.zeros((number_particles, 2))
    orientations = np.zeros((number_particles, 2))

    # set the orientation for all particles
    orientations[:, 0] = angle_x
    orientations[:, 1] = angle_y

    counter = 0
    # set the position of every particle
    for x in np.arange(length/2, x_length, length):
        positions[counter] = [x,x]
        counter += 1

    return positions, orientations