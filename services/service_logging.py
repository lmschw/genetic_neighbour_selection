import csv
import numpy as np

"""
Service containing methods to do with logging.
"""

def create_headers(len_c_values, is_best=False):
    """
    Creates the headers for the csv file.

    Params:
        - len_c_values (int): how many values are in the c_values
        - is_best (boolean): if the log is meant for the best result or for individual results

    Returns:
        A list containing all the headers.
    """
    if is_best:
        headers = ['iter']
    else:
        headers = ['iter', 'ind']
    individual_headers = [f"individual_{i}" for i in range(len_c_values)]
    headers.extend(individual_headers)
    headers.append('fitness')
    return headers

def initialise_log_file_with_headers(headers, save_path):
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)

def create_dicts_for_logging(iter, individuals, fitnesses):
    dict_list = []
    for ind in range(len(individuals)):
        dict_list.append(create_dict(iter, ind, individuals[ind], fitnesses[ind]))
    return prepare_individuals_for_csv_logging(dict_list)

def create_dict(iter, ind, c_values, fitness):
    return {'iter': iter, 'ind': ind, 'individual': c_values, 'fitness': fitness}

def prepare_individuals_for_csv_logging(dict_list):
    new_dict_list = []
    for dict in dict_list:
        new_dict = {}
        for k, v in dict.items():
            if isinstance(v, np.ndarray):
                for i in range(len(v)):
                    new_dict[f"individual_{i}"] = v[i]
            else:
                new_dict[k] = v
        new_dict_list.append(new_dict)
    return new_dict_list

def log_results_to_csv(dict_list, save_path, prepare=False):
    if prepare == True:
        dict_list = prepare_individuals_for_csv_logging(dict_list)
    with open(save_path, 'a', newline='') as f:
        w = csv.writer(f)
        for dict in dict_list:
            w.writerow(dict.values())