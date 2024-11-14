import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


from event.enums_event import EventEffect
from event.enums_event import DistributionType
from event.enums_event import EventSelectionType
from event.base_event import BaseEvent

import services.service_orientations as sorient
import services.service_preparation as sprep
import services.service_helper as shelp

class ExternalStimulusOrientationChangeEvent(BaseEvent):
    # TODO refactor to allow areas with a radius bigger than the radius of a particle, i.e. remove neighbourCells and determine all affected cells here
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, start_timestep, duration, domain_size, event_effect, distribution_type, areas=None, radius=None, number_of_affected=None, event_selection_type=None, 
                 angle=None, noise_percentage=None):
        """
        Creates an external stimulus event that affects part of the swarm at a given timestep.

        Params:
            - start_timestep (int): the first timestep at which the stimulus is presented and affects the swarm
            - duration (int): the number of timesteps during which the stimulus is present and affects the swarm
            - domain_size (tuple of floats): the size of the domain
            - event_effect (Enumevent_effect): how the orientations should be affected
            - distribution_type (distribution_type): whether the event is global or local in nature
            - areas (list of arrays containing [x_center, y_center, radius]) [optional]: where the event is supposed to take effect. Required for Local events
            - radius (float) [optional]: the event radius
            - noise_percentage (float, range: 0-100) [optional]: how much noise is added to the orientation determined by the event (only works for certain events)
            
        Returns:
            No return.
        """
        super().__init__(start_timestep=start_timestep, duration=duration, domain_size=domain_size, event_effect=event_effect, noise_percentage=noise_percentage)
        self.angle = angle
        self.distribution_type = distribution_type
        self.areas = areas
        self.number_of_affected = number_of_affected
        self.event_selection_type = event_selection_type

        match self.distribution_type:
            case DistributionType.GLOBAL:
                self.radius = (domain_size[0] * domain_size[1]) /np.pi
            case DistributionType.LOCAL_SINGLE_SITE:
                self.radius = self.areas[0][2]

        if radius:
            self.radius = radius

        if self.distribution_type != DistributionType.GLOBAL and self.areas == None:
            raise Exception("Local effects require the area to be specified")
        
        if self.number_of_affected and self.radius:
            print("Radius is set. The full number of affected particles may not be reached.")
        
    def get_short_print_version(self):
        return f"t{self.start_timestep}d{self.duration}e{self.event_effect.val}a{self.angle}dt{self.distribution_type.value}a{self.areas}"

    def get_parameter_summary(self):
        summary = super().get_parameter_summary()
        summary["angle"] = self.angle
        summary["distribution_type"] = self.distribution_type.name
        summary["areas"] = self.areas
        summary["radius"] = self.radius
        summary["number_of_affected"] = self.number_of_affected
        if self.event_selection_type:
            summary["event_selection_type"] = self.event_selection_type.value
        return summary
    
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
            The orientations of all particles after the event has been executed.
        """
        pos_with_center = np.zeros((total_number_of_particles+1, 2))
        pos_with_center[:-1] = positions
        pos_with_center[-1] = self.get_origin_point()
        rij2 = sorient.get_differences(pos_with_center, self.domain_size)
        relevant_distances = rij2[-1][:-1] # only the comps to the origin and without the origin point
        candidates = (relevant_distances <= self.radius**2)
        affected = self.selectAffected(candidates, relevant_distances)

        match self.event_effect:
            case EventEffect.ALIGN_TO_FIXED_ANGLE:
                orientations[affected] = sorient.compute_uv_coordinates(self.angle)
            case EventEffect.ALIGN_TO_FIXED_ANGLE_NOISE:
                orientations[affected] = sorient.compute_uv_coordinates(self.angle)
                orientations[affected] = self.apply_noise_distribution(orientations[affected])
            case EventEffect.AWAY_FROM_ORIGIN:
                orientations[affected] = self.compute_away_from_origin(positions[affected])
            case EventEffect.RANDOM:
                orientations[affected] = self.__get_random_orientations(np.count_nonzero(affected))
        orientations = sorient.normalize_orientations(orientations)

        return orientations # external events do not directly impact the values
    

    def selectAffected(self, candidates, rij2):
        """
        Determines which particles are affected by the event.

        Params:
            - candidates (array of boolean): which particles are within range, i.e. within the event radius
            - rij2 (array of floats): the distance squared of every particle to the event focus point

        Returns:
            Array of booleans representing which particles are affected by the event.
        """
        if self.number_of_affected == None:
            number_of_affected = len(candidates.nonzero()[0])
        else:
            number_of_affected = self.number_of_affected

        number_of_affected = int(number_of_affected)

        preselection = candidates # default case, we take all the candidates
        match self.event_selection_type:
            case EventSelectionType.NEAREST_DISTANCE:
                indices = np.argsort(rij2)[:number_of_affected]
                preselection = np.full(len(candidates), False)
                preselection[indices] = True
            case EventSelectionType.RANDOM:
                indices = candidates.nonzero()[0]
                selected_indices = np.random.choice(indices, number_of_affected, replace=False)
                preselection = np.full(len(candidates), False)
                preselection[selected_indices] = True
        return candidates & preselection

    def compute_away_from_origin(self, positions):
        """
        Computes the (u,v)-coordinates for the orientation after turning away from the point of origin.

        Params:
            - position ([X,Y]): the position of the current particle that should turn away from the point of origin

        Returns:
            [U,V]-coordinates representing the new orientation of the current particle.
        """
        angles = self.__compute_angle_with_regard_to_origin(positions)
        return sorient.compute_uv_coordinates_for_list(angles)

    def __compute_angle_with_regard_to_origin(self, positions):
        """
        Computes the angle between the position of the current particle and the point of origin of the event.

        Params:
            - position ([X,Y]): the position of the current particle that should turn towards the point of origin

        Returns:
            The angle in radians between the two points.
        """
        orientation_from_origin = positions - self.get_origin_point()
        anglesRadian = sorient.compute_angles_for_orientations(orientation_from_origin)
        return anglesRadian

    def get_origin_point(self):
        """
        Determines the point of origin of the event.

        Params:
            None

        Returns:
            The point of origin of the event in [X,Y]-coordinates.
        """
        match self.distribution_type:
            case DistributionType.GLOBAL:
                origin = (self.domain_size[0]/2, self.domain_size[1]/2)
            case DistributionType.LOCAL_SINGLE_SITE:
                origin = self.areas[0][:2]
        return origin

    
    def __get_random_orientations(self, num_affected_particles):
        """
        Selects a random orientation.

        Params:
            - num_affected_particles (int): the number of particles affected by the event

        Returns:
            A random orientation in [U,V]-coordinates.
        """
        return sorient.normalize_orientations(np.random.rand(num_affected_particles, len(self.domain_size))-0.5)