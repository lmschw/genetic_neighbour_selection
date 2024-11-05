import matplotlib.pyplot as plt

class MatplotlibAnimator:
    """
    The animator instance driven by Matplotlib.
    """
    
    def __init__(self, simulation_data, domain_size, colours=None, red_indices=[]):
        """
        Parameters:
            - simulation_data (array of arrays): The simulation data array containing times, positions and orientations
            - domain_size (tuple of floats): The tuple that represents the lenghts of the square domain in each dimension.
            - colours (array of arrays) [optional]: Contains the colour for every particle at every timestep. By default, all particles will be shown in black
            - red_indices (list) [optional]: A list containing indices of particles that will be shown in red. Will be ignored if a colour array is passed
        """
        self._simulation_data = simulation_data
        self._domain_size = domain_size
        self._colours = colours
        self._red_indices = red_indices

        self._initialize()

    def prepare(self, animator, frames=100, frame_interval=10):
        """
        Prepares the appropriate animator.

        Parameters:
            - animator (Animator): The appropriate animator class.

        Returns:
            Prepared animator feeded with simulation data.
        """
        prepared_animator =  animator.prepare_animation(self._figure, frames, frame_interval)

        return prepared_animator.set_simulation_data(self._simulation_data, self._domain_size, self._colours, self._red_indices)

    def _initialize(self):
        """
        Initializes matplotlib for animation.

        Parameters:
        None
        
        Returns:
            plt.figure()
        """
        self._figure = plt.figure()
        
        return self._figure