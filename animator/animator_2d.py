import matplotlib.pyplot as plt
import matplotlib.patches as patches
from animator.animator import Animator

class Animator2D(Animator):
    """
    Animator class for 2D graphical representation.
    """

    def __init__(self, model_params):
        self.set_params(model_params)

    def _animate(self, i):
        """
        Animator class that goes through sim data.

        Parameters:
            i (int): Loop index.

        Returns:
            Nothing
        """

        if i % 500 == 0:
            print(i)

        plt.clf()

        plt.quiver(self._positions[i,:,0],self._positions[i,:,1],self._orientations[i,:,0],self._orientations[i,:,1],color=self._colours[i])
        plt.xlim(0,self._domain_size[0])
        plt.ylim(0,self._domain_size[1])
        plt.title(f"$t$={self._time[i]:.2f}")
        