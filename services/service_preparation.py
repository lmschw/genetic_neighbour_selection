import numpy as np

def get_noise_amplitude_value_for_percentage(percentage):
    """
    Paramters:
        - percentage (int, 1-100)
    """
    return 2 * np.pi * (percentage/100)

def get_number_of_particles_for_constant_density(density, domain_size):
    """
    Computes the number of particles to keep the density constant for the supplied domain size.
    Density formula: "density" = "number of particles" / "domain area"

    Parameters:
        - density (float): the desired constant density of the domain
        - domainSize (tuple): tuple containing the x and y dimensions of the domain size

    Returns:
        The number of particles to be placed in the domain that corresponds to the density.
    """
    return int(density * (domain_size[0] * domain_size[1])) # density * area

def get_density(domain_size, number_of_particles):
    """
    Computes the density of a given system.
    Density formula: "density" = "number of particles" / "domain area"

    Parameters:
        - domainSize (tuple): tuple containing the x and y dimensions of the domain size
        - numberOfParticles (int): the number of particles to be placed in the domain

    Returns:
        The density of the system as a float.
    """
    return number_of_particles / (domain_size[0] * domain_size[1]) # n / area


def get_domain_size_for_constant_density(density, number_particles):
    """
    Computes the domain size to keep the density constant for the supplied number of particles.
    Density formula: "density" = "number of particles" / "domain area"

    Parameters:
        - density (float): the desired constant density of the domain
        - numberOfParticles (int): the number of particles to be placed in the domain

    Returns:
        A tuple containing the x and y dimensions of the domain size that corresponds to the density.
    """
    area = number_particles / density
    return (np.sqrt(area), np.sqrt(area))