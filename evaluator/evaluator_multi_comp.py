import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd

from evaluator.evaluator import Evaluator

# matplotlib default colours with corresponding colours that are 65% and 50% lighter
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
BACKGROUND_COLOURS_65_PERCENT_LIGHTER = ['#a6d1f0', '#ffd2ab', '#abe8ab', '#f1b3b3', '#dacae8',
                                         '#dbc1bc', '#f5cfea', '#d2d2d2', '#eff0aa', '#a7eef5']
BACKGROUND_COLOURS_50_PERCENT_LIGHTER = ['#7fbee9', '#ffbf86', '#87de87', '#eb9293', '#c9b3de',
                                         '#cca69f', '#f1bbe0', '#bfbfbf', '#e8e985', '#81e7f1']
BACKGROUND_COLOURS = BACKGROUND_COLOURS_50_PERCENT_LIGHTER
class EvaluatorMultiAvgComp(object):
    """
    Implementation of the evaluation mechanism for the Vicsek model for comparison of multiple models.
    """

    def __init__(self, model_params, simulation_data=None, evaluation_timestep_interval=1):
        """
        Initialises the evaluator.

        Parameters:
            - model_params (array of dictionaries): contains the model parameters for each model
            - metric (EnumMetrics.Metrics) [optional]: the metric according to which the models' performances should be evaluated
            - simulation_data (array of (time array, positions array, orientation array, colours array)) [optional]: contains all the simulation data for each model
            - evaluation_timestep_interval (int) [optional]: the interval of the timesteps to be evaluated. By default, every time step is evaluated
        
        Returns:
            Nothing.
        """
        self.simulation_data = simulation_data
        self.model_params = model_params
        self.evaluation_timestep_interval = evaluation_timestep_interval

    def evaluate(self):
        """
        Evaluates all models according to the metric specified for the evaluator.

        Parameters:
            None

        Returns:
            A dictionary with the results for each model at every time step.
        """
        dd = defaultdict(list)
        variance_data = []
        for model in range(len(self.simulation_data)):
            variance_data_model = []
            #print(f"evaluating {model}/{len(self.simulation_data)}")
            results = []
            for individual_run in range(len(self.simulation_data[model])):
                #print(f"step {individual_run}/{len(self.simulation_data[model])}")
                evaluator = Evaluator(self.model_params[model][individual_run], self.simulation_data[model][individual_run], self.evaluation_timestep_interval)
                result = evaluator.evaluate()
                results.append(result)
            
            ddi = defaultdict(list)
            for d in results: 
                for key, value in d.items():
                    ddi[key].append(value)
            
            for m in range(len(ddi)):
                idx = m * self.evaluation_timestep_interval
                dd[idx].append(np.average(ddi[idx]))
                variance_data_model.append(np.array(ddi[idx]))

            variance_data.append(variance_data_model)
        return dd, variance_data

    
    def visualize(self, data, labels, x_label=None, y_label=None, subtitle=None, colour_background_for_timesteps=None, variance_data=None, xlim=None, ylim=None, save_path=None):
        """
        Visualizes and optionally saves the results of the evaluation as a graph.

        Parameters:
            - data (dictionary): a dictionary with the time step as key and an array of each model's result as values
            - labels (array of strings): the label for each model
            - x_label (string) [optional]: the label for the x-axis
            - y_label (string) [optional]: the label for the y-axis
            - subtitle (string) [optional]: subtitle to be included in the title of the visualisation
            - colour_background_for_timesteps ([start, stop]) [optional]: the start and stop timestep for the background colouring for the event duration
            - variance_data (array) [optional]: the variance for every timestep
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot
            - save_path (string) [optional]: the location and name of the file where the model should be saved. Will not be saved unless a save_path is provided

        Returns:
            Nothing.
        """


        if ylim == None:
            ylim = (0, 1.1)
        self.__create_standard_lineplot(data, labels, xlim=xlim, ylim=ylim)

        ax = plt.gca()
        # reset axis to start at (0.0)
        xlim = ax.get_xlim()
        ax.set_xlim((0, xlim[1]))
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))

        if variance_data != None:
            xlim = ax.get_xlim()
            x = np.arange(start=0, stop=len(variance_data[0]), step=1)
            for i in range(len(variance_data)):
                ax.fill_between(x, np.mean(variance_data[i], axis=1) - np.std(variance_data[i], axis=1), np.mean(variance_data[i], axis=1) + np.std(variance_data[i], axis=1), color=COLOURS[i], alpha=0.2)

        if x_label != None:
            plt.xlabel(x_label)
        if y_label != None:
            plt.ylabel(y_label)
        if subtitle != None:
            plt.title(f"""{subtitle}""")
        if not any(ele is None for ele in colour_background_for_timesteps):
            ax = plt.gca()
            ylim = ax.get_ylim()
            y = np.arange(ylim[0], ylim[1], 0.01)
            ax.fill_betweenx(y, colour_background_for_timesteps[0], colour_background_for_timesteps[1], facecolor='green', alpha=0.2)
        if save_path != None:
            plt.savefig(save_path)
        #plt.show()
        plt.close()

    def evaluate_and_visualize(self, labels, x_label=None, y_label=None, subtitle=None, colour_background_for_timesteps=(None,None), show_variance=False, xlim=None, ylim=None, save_path=None):
        """
        Evaluates and subsequently visualises the results for multiple models.

        Parameters:
            - labels (array of strings): the label for each model
            - x_label (string) [optional]: the label for the x-axis
            - y_label (string) [optional]: the label for the y-axis
            - subtitle (string) [optional]: subtitle to be included in the title of the visualisation
            - colour_background_for_timesteps ([start, stop]) [optional]: the start and stop timestep for the background colouring for the event duration
            - show_variance (boolean) [optional]: whether the variance data should be added to the plot, by default False
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot
            - save_path (string) [optional]: the location and name of the file where the model should be saved. Will not be saved unless a save_path is provided

        Returns:
            Nothing.
        """
        data, variance_data = self.evaluate()
        if show_variance == False:
            variance_data = None
        self.visualize(data, labels, x_label=x_label, y_label=y_label, subtitle=subtitle, colour_background_for_timesteps=colour_background_for_timesteps, variance_data=variance_data, xlim=xlim, ylim=ylim, save_path=save_path)
        
    def __create_standard_lineplot(self, data, labels, xlim=None, ylim=None):
        """
        Creates a line plot for every model at every timestep

        Parameters:
            - data (dictionary): a dictionary with the time step as its key and a list of the number of clusters for every model as its value
            - labels (list of strings): labels for the models
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot

        Returns:
            Nothing.
        """
        sorted(data.items())
        df = pd.DataFrame(data, index=labels).T

        if xlim != None and ylim != None:
            df.plot.line(xlim=xlim, ylim=ylim)
        elif xlim != None:
            df.plot.line(xlim=xlim)
        elif ylim != None:
            df.plot.line(ylim=ylim)
        else:
            df.plot.line()