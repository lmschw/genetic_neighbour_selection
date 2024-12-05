
import numpy as np
import pandas as pd
import os

import services.service_logging as slog

def delete_csv_file(filepath):
    if(os.path.exists(filepath) and os.path.isfile(filepath)):
        os.remove(filepath)
    else:
        raise Exception(f"could not delete file: {filepath}")

def test_create_headers_empty():
    len_c_values = 0
    expected = ['iter', 'ind', 'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values)
    equal = True
    assert len(expected) == len(result), "test_create_headers_empty: The number of headers should be 4"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

def test_create_headers_full_ranking():
    # without own and random with distinction
    n = 5
    len_c_values = (n-1) * 3
    expected = ['iter', 'ind', 
                'o_0', 'o_1', 'o_2', 'o_3', 
                'p_0', 'p_1', 'p_2', 'p_3', 
                'b_0', 'b_1', 'b_2', 'b_3', 
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[True, True, True], 
                                 has_own=False, has_random=False, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_full_ranking: The number of headers should be equal (no own, no random, with distinction)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # without own and random without distinction
    expected = ['iter', 'ind', 
                'individual_0', 'individual_1', 'individual_2', 'individual_3', 
                'individual_4', 'individual_5', 'individual_6', 'individual_7', 
                'individual_8', 'individual_9', 'individual_10', 'individual_11', 
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, ranking_by=[True, True, True], 
                                 has_own=False, has_random=False, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_full_ranking: The number of headers should be equal (no own, no random, without distinction)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # with own with distinction
    expected = ['iter', 'ind', 
                'o_0', 'o_1', 'o_2', 'o_3', 
                'p_0', 'p_1', 'p_2', 'p_3', 
                'b_0', 'b_1', 'b_2', 'b_3',
                'own', 
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[True, True, True], 
                                 has_own=True, has_random=False, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_full_ranking: The number of headers should be equal (own, no random, with distinction)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # with own, no distinction
    expected = ['iter', 'ind', 
                'individual_0', 'individual_1', 'individual_2', 'individual_3', 
                'individual_4', 'individual_5', 'individual_6', 'individual_7', 
                'individual_8', 'individual_9', 'individual_10', 'individual_11', 
                'own', 'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, ranking_by=[True, True, True], 
                                 has_own=True, has_random=False, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_full_ranking: The number of headers should be equal (own, no random, without distinction)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # with random with distinction
    expected = ['iter', 'ind', 
                'o_0', 'o_1', 'o_2', 'o_3', 
                'p_0', 'p_1', 'p_2', 'p_3', 
                'b_0', 'b_1', 'b_2', 'b_3',
                'rand', 
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[True, True, True], 
                                 has_own=False, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_full_ranking: The number of headers should be equal (no own, random, with distinction)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # with random, no distinction
    expected = ['iter', 'ind', 
                'individual_0', 'individual_1', 'individual_2', 'individual_3', 
                'individual_4', 'individual_5', 'individual_6', 'individual_7', 
                'individual_8', 'individual_9', 'individual_10', 'individual_11', 
                'rand', 'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, ranking_by=[True, True, True], 
                                 has_own=False, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_full_ranking: The number of headers should be equal (no own, random, without distinction)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # with own and random with distinction
    expected = ['iter', 'ind', 
                'o_0', 'o_1', 'o_2', 'o_3', 
                'p_0', 'p_1', 'p_2', 'p_3', 
                'b_0', 'b_1', 'b_2', 'b_3',
                'own', 'rand',
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[True, True, True], 
                                 has_own=True, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_full_ranking: The number of headers should be equal (own, random, with distinction)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # with own, no distinction
    expected = ['iter', 'ind', 
                'individual_0', 'individual_1', 'individual_2', 'individual_3', 
                'individual_4', 'individual_5', 'individual_6', 'individual_7', 
                'individual_8', 'individual_9', 'individual_10', 'individual_11', 
                'own', 'rand', 'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, ranking_by=[True, True, True], 
                                 has_own=True, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_full_ranking: The number of headers should be equal (own, random, without distinction)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

def test_create_headers_partial_ranking():
    # orientation only
    len_c_values = 4
    n = 5
    expected = ['iter', 'ind', 
                'o_0', 'o_1', 'o_2', 'o_3', 
                'own', 'rand',
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[True, False, False], 
                                 has_own=True, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_partial_ranking: The number of headers should be equal (orientation only)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # distance only
    expected = ['iter', 'ind', 
                'p_0', 'p_1', 'p_2', 'p_3', 
                'own', 'rand',
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[False, True, False], 
                                 has_own=True, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_partial_ranking: The number of headers should be equal (distance only)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # bearing only
    expected = ['iter', 'ind', 
                'b_0', 'b_1', 'b_2', 'b_3',
                'own', 'rand',
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[False, False, True], 
                                 has_own=True, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_partial_ranking: The number of headers should be equal (bearing only)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # orientation and distance
    expected = ['iter', 'ind', 
                'o_0', 'o_1', 'o_2', 'o_3', 
                'p_0', 'p_1', 'p_2', 'p_3', 
                'own', 'rand',
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[True, True, False], 
                                 has_own=True, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_partial_ranking: The number of headers should be equal (orientation and distance)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # orientation and bearing
    expected = ['iter', 'ind', 
                'o_0', 'o_1', 'o_2', 'o_3', 
                'b_0', 'b_1', 'b_2', 'b_3',
                'own', 'rand',
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[True, False, True], 
                                 has_own=True, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_partial_ranking: The number of headers should be equal (orientation and bearing)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # distance and bearing
    expected = ['iter', 'ind', 
                'p_0', 'p_1', 'p_2', 'p_3', 
                'b_0', 'b_1', 'b_2', 'b_3',
                'own', 'rand',
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[False, True, True], 
                                 has_own=True, has_random=True, is_best=False)
    equal = True
    assert len(expected) == len(result), "test_create_headers_partial_ranking: The number of headers should be equal (distance and bearing)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

def test_create_headers_is_best():
    # empty
    len_c_values = 0
    expected = ['iter', 'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, is_best=True)
    equal = True
    assert len(expected) == len(result), "test_create_headers_is_best: The number of headers should be 3"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

    # full
    n = 5
    len_c_values = (n-1) * 3
    expected = ['iter', 
                'o_0', 'o_1', 'o_2', 'o_3', 
                'p_0', 'p_1', 'p_2', 'p_3', 
                'b_0', 'b_1', 'b_2', 'b_3',
                'own', 'rand',
                'fitness', 'fitness_order']
    result = slog.create_headers(len_c_values=len_c_values, n=n, ranking_by=[True, True, True], 
                                 has_own=True, has_random=True, is_best=True)
    equal = True
    assert len(expected) == len(result), "test_create_headers_is_best: The number of headers should be equal (full case)"
    for i in range(len(result)):
        if expected[i] != result[i]:
            equal = False
            break
    assert equal == True

def test_create_headers():
    # empty header
    test_create_headers_empty()
    # ranking full
    test_create_headers_full_ranking()
    # sub-ranking
    test_create_headers_partial_ranking()
    # is_best == True
    test_create_headers_is_best()

def test_initialise_log_file_with_headers():
    headers = ['iter', 'ind', 'o_0', 'o_1', 'p_0', 'p_1', 'b_0', 'b_1', 'fitness', 'fitness_order']
    filepath = "test_initialise_log_file_with_headers.csv"
    slog.initialise_log_file_with_headers(headers=headers, save_path=filepath)
    df = pd.read_csv(filepath)
    assert np.count_nonzero(df.columns == headers) == 10
    delete_csv_file(filepath=filepath)

def test_create_dicts_for_logging():
    individuals = [np.array([0, 0.2, 0.4, 0.6]),
                   np.array([0.1, 0.3, 0.5, 0.7])]
    fitnesses = [0.001, 0.002]
    fitnesses_order = [0.0001, 0.0002]
    result = slog.create_dicts_for_logging(iter=0, individuals=individuals, fitnesses=fitnesses, fitnesses_order=fitnesses_order, n=3, ranking_by=[True, True, False])
    assert result[0]['iter'] == 0, "test_create_dicts_for_logging: Iteration for first individual should be 0"
    assert result[0]['ind'] == 0, "test_create_dicts_for_logging: First individual should be 0"
    assert result[0]['o_0'] == individuals[0][0], "test_create_dicts_for_logging: c_values have not been added properly to first individual (o_0)"
    assert result[0]['o_1'] == individuals[0][1], "test_create_dicts_for_logging: c_values have not been added properly to first individual (o_1)"
    assert result[0]['p_0'] == individuals[0][2], "test_create_dicts_for_logging: c_values have not been added properly to first individual (p_0)"
    assert result[0]['p_1'] == individuals[0][3], "test_create_dicts_for_logging: c_values have not been added properly to first individual (p_1)"
    assert result[0]['fitness'] == fitnesses[0], "test_create_dicts_for_logging: The fitness for the first individual is incorrect"
    assert result[0]['fitness_order'] == fitnesses_order[0], "test_create_dicts_for_logging: The fitness_order for the first individual is incorrect"
    assert result[1]['iter'] == 0, "test_create_dicts_for_logging: Iteration for second individual should be 0"
    assert result[1]['ind'] == 1, "test_create_dicts_for_logging: Second individual should be 1"
    assert result[1]['o_0'] == individuals[1][0], "test_create_dicts_for_logging: c_values have not been added properly to second individual (o_0)"
    assert result[1]['o_1'] == individuals[1][1], "test_create_dicts_for_logging: c_values have not been added properly to second individual (o_1)"
    assert result[1]['p_0'] == individuals[1][2], "test_create_dicts_for_logging: c_values have not been added properly to second individual (p_0)"
    assert result[1]['p_1'] == individuals[1][3], "test_create_dicts_for_logging: c_values have not been added properly to second individual (p_1)"
    assert result[1]['fitness'] == fitnesses[1], "test_create_dicts_for_logging: The fitness for the second individual is incorrect"
    assert result[1]['fitness_order'] == fitnesses_order[1], "test_create_dicts_for_logging: The fitness_order for the second individual is incorrect"

def test_create_dict():
    iter = 1
    ind = 7
    c_values = np.array([0.1, 0.2])
    fitness = 0.001
    fitness_order = 0.0001
    result = slog.create_dict(iter=iter, ind=ind, c_values=c_values, fitness=fitness, fitness_order=fitness_order)
    assert result['iter'] == iter, "test_create_dict: The iteration is incorrect"
    assert result['ind'] == ind, "test_create_dict: The ind number is set incorrectly"
    assert result['individual'][0] == c_values[0], "test_create_dict: The c_values have been set incorrectly"
    assert result['individual'][1] == c_values[1], "test_create_dict: The c_values have been set incorrectly"
    assert result['fitness'] == fitness, "test_create_dict: The fitness value is incorrect"
    assert result['fitness_order'] == fitness_order, "test_create_dict: The fitness_order value is incorrect"

def test_prepare_individuals_for_csv_logging():
    data = [{'iter': 0, 'ind': 0, 'individual': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 'fitness': 0.001, 'fitness_order': 0.0001},
            {'iter': 2, 'ind': 7, 'individual': np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66]), 'fitness': 0.01, 'fitness_order': 0.001}]
    result = slog.prepare_individuals_for_csv_logging(dict_list=data, ranking_by=[True, True, True], n=3)
    assert result[0]['iter'] == data[0]['iter'], "test_create_dicts_for_logging: Iteration for first individual should be 0"
    assert result[0]['ind'] == data[0]['ind'], "test_create_dicts_for_logging: First individual should be 0"
    assert result[0]['o_0'] == data[0]['individual'][0], "test_create_dicts_for_logging: c_values have not been added properly to first individual (o_0)"
    assert result[0]['o_1'] == data[0]['individual'][1], "test_create_dicts_for_logging: c_values have not been added properly to first individual (o_1)"
    assert result[0]['p_0'] == data[0]['individual'][2], "test_create_dicts_for_logging: c_values have not been added properly to first individual (p_0)"
    assert result[0]['p_1'] == data[0]['individual'][3], "test_create_dicts_for_logging: c_values have not been added properly to first individual (p_1)"
    assert result[0]['b_0'] == data[0]['individual'][4], "test_create_dicts_for_logging: c_values have not been added properly to first individual (b_0)"
    assert result[0]['b_1'] == data[0]['individual'][5], "test_create_dicts_for_logging: c_values have not been added properly to first individual (b_1)"
    assert result[0]['fitness'] == data[0]['fitness'], "test_create_dicts_for_logging: The fitness for the first individual is incorrect"
    assert result[0]['fitness_order'] == data[0]['fitness_order'], "test_create_dicts_for_logging: The fitness_order for the first individual is incorrect"
    assert result[1]['iter'] == data[1]['iter'], "test_create_dicts_for_logging: Iteration for first individual should be 2"
    assert result[1]['ind'] == data[1]['ind'], "test_create_dicts_for_logging: First individual should be 7"
    assert result[1]['o_0'] == data[1]['individual'][0], "test_create_dicts_for_logging: c_values have not been added properly to first individual (o_0)"
    assert result[1]['o_1'] == data[1]['individual'][1], "test_create_dicts_for_logging: c_values have not been added properly to first individual (o_1)"
    assert result[1]['p_0'] == data[1]['individual'][2], "test_create_dicts_for_logging: c_values have not been added properly to first individual (p_0)"
    assert result[1]['p_1'] == data[1]['individual'][3], "test_create_dicts_for_logging: c_values have not been added properly to first individual (p_1)"
    assert result[1]['b_0'] == data[1]['individual'][4], "test_create_dicts_for_logging: c_values have not been added properly to first individual (b_0)"
    assert result[1]['b_1'] == data[1]['individual'][5], "test_create_dicts_for_logging: c_values have not been added properly to first individual (b_1)"
    assert result[1]['fitness'] == data[1]['fitness'], "test_create_dicts_for_logging: The fitness for the first individual is incorrect"
    assert result[1]['fitness_order'] == data[1]['fitness_order'], "test_create_dicts_for_logging: The fitness_order for the first individual is incorrect"


def test_log_results_to_csv():
    headers = ['iter', 'ind', 'o_0', 'o_1', 'p_0', 'p_1', 'b_0', 'b_1', 'fitness', 'fitness_order']
    unprepared_data = [{'iter': 0, 'ind': 0, 'individual': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 'fitness': 0.001, 'fitness_order': 0.0001},
                     {'iter': 2, 'ind': 7, 'individual': np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66]), 'fitness': 0.01, 'fitness_order': 0.001}]
    prepared_data = [{'iter': 1, 'ind': 1, 'o_0': 0.12, 'o_1': 0.21, 'p_0': 0.31, 'p_1': 0.41, 'b_0': 0.51, 'b_1': 0.61, 'fitness': 0.0011, 'fitness_order': 0.00011},
                    {'iter': 3, 'ind': 8, 'o_0': 0.13, 'o_1': 0.22, 'p_0': 0.32, 'p_1': 0.42, 'b_0': 0.52, 'b_1': 0.62, 'fitness': 0.012, 'fitness_order': 0.0012}]
    filepath = "test_initialise_log_file_with_headers.csv"
    slog.initialise_log_file_with_headers(headers=headers, save_path=filepath)
    slog.log_results_to_csv(dict_list=unprepared_data, save_path=filepath, ranking_by=[True, True, True], n=3, prepare=True)
    df = pd.read_csv(filepath)
    assert np.count_nonzero(df['iter'] == [0,2]) == 2
    assert np.count_nonzero(df['ind'] == [0,7]) == 2
    assert np.count_nonzero(df['o_0'] == [0.1,0.11]) == 2
    assert np.count_nonzero(df['o_1'] == [0.2,0.22]) == 2
    assert np.count_nonzero(df['p_0'] == [0.3,0.33]) == 2
    assert np.count_nonzero(df['p_1'] == [0.4,0.44]) == 2
    assert np.count_nonzero(df['b_0'] == [0.5,0.55]) == 2
    assert np.count_nonzero(df['b_1'] == [0.6,0.66]) == 2
    assert np.count_nonzero(df['fitness'] == [0.001,0.01]) == 2
    assert np.count_nonzero(df['fitness_order'] == [0.0001,0.001]) == 2
    delete_csv_file(filepath=filepath)

    slog.initialise_log_file_with_headers(headers=headers, save_path=filepath)
    slog.log_results_to_csv(dict_list=prepared_data, save_path=filepath, ranking_by=[True, True, True], n=3, prepare=False)
    df = pd.read_csv(filepath)
    assert np.count_nonzero(df['iter'] == [1,3]) == 2
    assert np.count_nonzero(df['ind'] == [1,8]) == 2
    assert np.count_nonzero(df['o_0'] == [0.12,0.13]) == 2
    assert np.count_nonzero(df['o_1'] == [0.21,0.22]) == 2
    assert np.count_nonzero(df['p_0'] == [0.31,0.32]) == 2
    assert np.count_nonzero(df['p_1'] == [0.41,0.42]) == 2
    assert np.count_nonzero(df['b_0'] == [0.51,0.52]) == 2
    assert np.count_nonzero(df['b_1'] == [0.61,0.62]) == 2
    assert np.count_nonzero(df['fitness'] == [0.0011,0.012]) == 2
    assert np.count_nonzero(df['fitness_order'] == [0.00011,0.0012]) == 2
    delete_csv_file(filepath=filepath)

def run_all():
    test_create_headers()
    test_initialise_log_file_with_headers()
    test_create_dicts_for_logging()
    test_create_dict()
    test_prepare_individuals_for_csv_logging()
    test_log_results_to_csv()
    print("Everything passed for service_logging")