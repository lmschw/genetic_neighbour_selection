
import numpy as np

import services.service_preparation as sprep
import services.service_helper as shelp

class BaseEvent:
    # TODO refactor to allow areas with a radius bigger than the radius of a particle, i.e. remove neighbourCells and determine all affected cells here
    # TODO make noise_percentage applicable to all events
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, start_timestep, duration, domain_size, event_effect, noise_percentage=None):
        """
        Creates an event that affects part of the swarm at a given timestep.

        Params:
            - start_timestep (int): the first timestep at which the stimulus is presented and affects the swarm
            - duration (int): the number of timesteps during which the stimulus is present and affects the swarm
            - domain_size (tuple of floats): the size of the domain
            - event_effect (Enumevent_effect): how the orientations should be affected
            - noise_percentage (float, range: 0-100) [optional]: how much noise is added to the orientation determined by the event (only works for certain events)
            
        Returns:
            No return.
        """
        self.start_timestep = start_timestep
        self.duration = duration
        self.event_effect = event_effect
        self.domain_size = np.asarray(domain_size)
        self.noise_percentage = noise_percentage
        if self.noise_percentage != None:
            self.noise = sprep.get_noise_amplitude_value_for_percentage(self.noise_percentage)


    def get_short_print_version(self):
        return f"t{self.start_timestep}d{self.duration}e{self.event_effect.val}"

    def get_parameter_summary(self):
        summary = {"start_timestep": self.start_timestep,
            "duration": self.duration,
            "event_effect": self.event_effect.val,
            "domain_size": self.domain_size.tolist(),
            "noise_percentage": self.noise_percentage,
            }
        return summary

    def check(self, total_number_of_particles, current_timestep, positions, orientations):
        """
        Checks if the event is triggered at the current timestep and executes it if relevant.

        Params:
            - total_number_of_particles (int): the total number of particles within the domain. Used to compute the number of affected particles
            - current_timestep (int): the timestep within the experiment run to see if the event should be triggered
            - positions (array of tuples (x,y)): the position of every particle in the domain at the current timestep
            - orientations (array of tuples (u,v)): the orientation of every particle in the domain at the current timestep
            - nsms (array of NeighbourSelectionMechanisms): the neighbour selection mechanism currently selected by each individual at the current timestep
            - ks (array of int): the number of neighbours currently selected by each individual at the current timestep
            - speeds (array of float): the speed of every particle at the current timestep
            - dt (float) [optional]: the difference between the timesteps
            - activationTimeDelays (array of int) [optional]: the time delay for the updates of each individual
            - isActivationTimeDelayRelevantForEvent (boolean) [optional]: whether the event can affect particles that may not be ready to update due to a time delay. They may still be selected but will retain their current values
            - colourType (ColourType) [optional]: if and how particles should be encoded for colour for future video rendering

        Returns:
            The orientations of all particles - altered if the event has taken place, unaltered otherwise.
        """

        if self.check_timestep(current_timestep):
            # if current_timestep == self.start_timestep or current_timestep == (self.start_timestep + self.duration):
            #     print(f"executing event at timestep {current_timestep}")
            orientations = self.execute_event(total_number_of_particles=total_number_of_particles, positions=positions, orientations=orientations)
        return orientations

    def check_timestep(self, current_timestep):
        """
        Checks if the event should be triggered.

        Params:
            - current_timestep (int): the timestep within the experiment run to see if the event should be triggered

        Returns:
            A boolean representing whether or not the event should be triggered.
        """
        return self.start_timestep <= current_timestep and current_timestep <= (self.start_timestep + self.duration)
    
    def apply_noise_distribution(self, orientations):
        """
        Applies noise to the orientations.

        Params:
            - orientations (array of tuples (u,v)): the orientation of every particle at the current timestep

        Returns:
            An array of tuples (u,v) that represents the orientation of every particle at the current timestep after noise has been applied.
        """
        return orientations + np.random.normal(scale=self.noise, size=(len(orientations), len(self.domain_size)))
    
    def execute_event(self, total_number_of_particles, positions, orientations):
        """
        Executes the event.

        Params:
            - total_number_of_particles (int): the total number of particles within the domain. Used to compute the number of affected particles
            - positions (array of tuples (x,y)): the position of every particle in the domain at the current timestep
            - orientations (array of tuples (u,v)): the orientation of every particle in the domain at the current timestep
            - nsms (array of NeighbourSelectionMechanisms): the neighbour selection mechanism currently selected by each individual at the current timestep
            - ks (array of int): the number of neighbours currently selected by each individual at the current timestep
            - speeds (array of float): the speed of every particle at the current timestep
            - dt (float) [optional]: the difference between the timesteps
            - colourType (ColourType) [optional]: if and how particles should be encoded for colour for future video rendering

        Returns:
            The orientations, neighbour selection mechanisms, ks, speeds, blockedness and colour of all particles after the event has been executed.
        """
        # base event does not do anything here
        return orientations