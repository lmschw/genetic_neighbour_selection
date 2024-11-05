import services.service_orientations as sorient

import numpy as np

class Evaluator(object):
    """
    Implementation of the evaluation mechanism for the Vicsek model for a single model.
    """
    def __init__(self, model_params, simulation_data=None, evaluation_timestep_interval=1):
        """
        Initialises the evaluator.

        Parameters:
            - model_params (array of dictionaries): contains the model parameters for the current model
            - simulation_data (array of (time array, positions array, orientation array, colours array)) [optional]: contains all the simulation data
            - evaluation_timestep_interval (int) [optional]: the interval of the timesteps to be evaluated. By default, every time step is evaluated
            - threshold (float) [optional]: the threshold for the AgglomerativeClustering cutoff
            - switchTypeValues (array of arrays of switchTypeValues) [optional]: the switch type value of every particle at every timestep
            - switchTypeOptions (tuple) [optional]: the two possible values for the switch type value
        
        Returns:
            Nothing.
        """
        if simulation_data != None:
            self.time, self.positions, self.orientations = simulation_data
        self.model_params = model_params
        self.evaluation_timestep_interval = evaluation_timestep_interval
        self.domain_size = np.array(model_params["domain_size"])

    def evaluate(self, start_timestep=0, end_timestep=None):
        """
        Evaluates the model according to the metric specified for the evaluator.

        Parameters:
            - start_timestep (int) [optional]: The first timestep used for the evaluation, i.e. the lower bound of the evaluation window. By default 0, the very first timestep
            - end_timestep (int) [optional]: The last timestep used for the evaluation, i.e. the upper bound of the evaluation window. By default the very last timestep

        Returns:
            A dictionary with the results for the model at every time step.
        """
        if len(self.time) < 1:
            print("ERROR: cannot evaluate without simulation_data. Please supply simulation_data, model_params and metric at Evaluator instantiation.")
            return
        if end_timestep == None:
            end_timestep = len(self.time)
        values_per_timestep = {}
        for i in range(len(self.time)):
            #if i % 100 == 0:
                #print(f"evaluating {i}/{len(self.time)}")
            if i % self.evaluation_timestep_interval == 0 and i >= start_timestep and i <= end_timestep:
                values_per_timestep[self.time[i]] = self.evaluate_single_timestep(orientations=self.orientations[i])
        return values_per_timestep
    
    def evaluate_single_timestep(self, orientations):
        return sorient.compute_global_order(orientations)