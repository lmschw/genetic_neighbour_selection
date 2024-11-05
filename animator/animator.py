from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

class Animator(object):
    """
    Animates the quiver plot for the Vicsek data
    """
    def prepare_animation(self, matplotlib_figure, frames=100, frame_interval = 10):
        """
        Prepares the 2D animator object for animation.

        Parameters:
            - matplotlib_figure (Figure): Matplotlibs figure object.
            - frame_interval (int): The interval between two frames.
            - frames (int): The number of frames used in the animation.

        Returns: 
            self
        """
        self._matplotlib_figure = matplotlib_figure
        self._frames = frames
        self._frame_interval = frame_interval

        return self

    def set_simulation_data(self, simulation_data, domain_size, colours=None, red_indices=[]):
        """
        Sets the simulation data.
        
        Parameters:
            - simulation_data (array of arrays): The simulation data array containing times, positions and orientations
            - domain_size (tuple of floats): The tuple that represents the lenghts of the square domain in each dimension.
            - colours (array of arrays) [optional]: Contains the colour for every particle at every timestep. By default, all particles will be shown in black
            - red_indices (list) [optional]: A list containing indices of particles that will be shown in red. Will be ignored if a colour array is passed

        Returns:
            self
        """        
        self._time, self._positions, self._orientations = simulation_data
        self._domain_size = domain_size

        if colours is None: # if the colours are provided, we don't mess with those as they may show an example
            a = np.array(len(self._positions[0]) * ['k'])
            if len(red_indices) > 0:
                a[red_indices] = 'r'
            self._colours = len(self._time) * [a.tolist()]
        else:
            self._colours = colours
        
        return self
    
    def set_parameters(self, n, k, noise, radius):
        self._n = n
        self._k = k
        self._noise = noise
        self._radius = radius

    def set_params(self, modelParams):
        self._n = modelParams["number_particles"]
        self._radius = modelParams["radius"]
        self._domain_size = modelParams["domain_size"]

    def show_animation(self):
        """
        Shows the animation

        Parameters:
            None

        Returns: 
            self
        """
        self._get_animation()
        plt.show()
        
        return self
    
    def save_animation(self, filename, fpsVar=25, codecVar="avi"):
        """
        Saves the animation. Requires FFMPEG

        Parameters:
            None

        Returns:
            Animator
        """
        print("Saving commenced...")
        animation = self._get_animation()
        animation.save(filename=filename, writer="ffmpeg")
        print("Saving completed.")
        #plt.close()
        return self
    
    def _get_animation(self):
        return self.animation if 'animation' in self.__dict__ else self._generate_animation()

    def _generate_animation(self):
        """
        Generates the animation.

        Parameters:
            None
        
        Returns
            animation object
        """
        self.animation = FuncAnimation(self._matplotlib_figure, self._animate, interval=self._frame_interval, frames = self._frames)

        return self.animation
