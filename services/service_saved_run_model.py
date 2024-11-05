#import csv

import codecs, json
import numpy as np
"""
Service contains static methods to save and load models to/from json files.
"""

def save_model(simulation_data, path="sample.json", model_params=None, saveInterval=1):
    """
    Saves a run model.

    Parameters:
        - simulation_data (times, positions, orientations): the data to be saved
        - path (string) [optional]: the location and name of the target file
        - model_params (dict) [optional]: a summary of the model's params such as n, k, neighbourSelectionMode etc.
        - saveInterval (int) [optional]: specifies the interval at which the saving should occur, i.e. if any time steps should be skipped
        
    Returns:
        Nothing. Creates or overwrites a file.
    """
    time, positions, orientations = simulation_data
    dict = {"time": __get_specified_intervals(saveInterval, time.tolist()), 
            "positions": __get_specified_intervals(saveInterval, positions.tolist()), 
            "orientations": __get_specified_intervals(saveInterval, orientations.tolist())}
    save_dict(path, dict, model_params)

def load_model(path):
    """
    Loads a single model from a single file.

    Parameters:
        - path (string): the location and file name of the file containing the model data

    Returns:
        The model's params as well as the simulation data containing the time, positions, orientations.
    """
    loadedJson = __load_json(path)

    model_params = loadedJson["model_params"]
    time = np.array(loadedJson["time"])
    positions = np.array(loadedJson["positions"])
    orientations = np.array(loadedJson["orientations"])
    return model_params, (time, positions, orientations)

def load_models(paths):
    """
    Loads multiple models from multiple files.

    Parameters:
        - paths (array of strings): An array containing the locations and names of the files containing a single model each
        
    Returns:
        Returns an array containing the model params for each model and a second array containing the simulation data for each model. Co-indexed.
    """
    data = []
    params = []
    for path in paths:
        model_params, simulation_data = load_model(path)
        params.append(model_params)
        data.append(simulation_data)
    return params, data
    
def __get_specified_intervals(interval, lst):
    """
    Selects the data within the list which coincides with the specified interval, e.g. every third data point.

    Parameters:
        - interval (int): which data points should be considered, e.g. 3 would indicate indices 0, 3, 6 etc.
        - lst (list): the data to be reduced according to the intervals
    
    Returns:
        A reduced list containing only the data points of the original list at the specified intervals.
    """
    return [lst[idx] for idx in range(0, len(lst)) if idx % interval == 0]

def save_dict(path, dict, model_params=None):
    """
    Saves the values of a dictionary to a file at the specified path.

    Parameters:
        - path (string): the location and name of the target file
        - dict (dictionary): the dictionary containing the data to be saved
        - model_params (dict) [optional]: a summary of the model's params such as n, k, neighbourSelectionMode etc.

    Returns:
        Nothing. Creates or overwrites a file.
    """
    if model_params != None:
        params_dict = {"model_params": model_params}
        params_dict.update(dict)
        dict = params_dict
        
    with open(path, "w") as outfile:
        json.dump(dict, outfile)

def __load_json(path):
    """
    Loads data as JSON from a single file.

    Parameters:
        - path (string): the location and file name of the file containing the data

    Returns:
        All the data from the file as JSON.
    """
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    return json.loads(obj_text)
