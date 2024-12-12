import numpy as np

import services.service_orientations as sorient

def test_normalize_orientations():
    orientations_empty = []
    orientations_mixed = [[1, 4], [3, -5], [0, 0], [1, 1], [-1, -1], [0.5, 0.5], [-0.5, -0.5]]

    expected_empty = []
    expected_mixed = [[0.24254, 0.97014], [0.5145, -0.85749], [0, 0], [0.70711, 0.70711], [-0.70711, -0.70711], [0.70711, 0.70711], [-0.70711, -0.70711]]

    result_empty = sorient.normalize_orientations(orientations=orientations_empty)
    result_mixed = sorient.normalize_orientations(orientations=orientations_mixed)

    assert len(expected_empty) == len(result_empty)
    assert len(expected_mixed) == len(result_mixed)
    equal = True
    for i in range(len(expected_mixed)):
        for j in range(2):
            if np.absolute(expected_mixed[i][j] - result_mixed[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

def test_normalize_angles():
    angles_empty = []
    angles_within_2_pi = np.array([np.pi, 0, 0.75*np.pi, 1.8*np.pi])
    angles_outside_2_pi = np.array([-np.pi, -0.1, 2.1*np.pi, 6*np.pi])

    expected_empty = []
    expected_within_2_pi = [np.pi, 0, 0.75*np.pi, 1.8 * np.pi]
    expected_outside_2_pi = [np.pi, 2*np.pi-0.1, 0.1*np.pi, 0]

    result_empty = sorient.normalize_angles(angles=angles_empty)
    result_within_2_pi = sorient.normalize_angles(angles=angles_within_2_pi)
    result_outside_2_pi = sorient.normalize_angles(angles=angles_outside_2_pi)

    assert len(expected_empty) == len(result_empty)
    assert len(expected_within_2_pi) == len(result_within_2_pi)
    equal = True
    for i in range(len(expected_within_2_pi)):
        if np.absolute(expected_within_2_pi[i] - result_within_2_pi[i]) > 0.01:
            equal = False
            break
    assert equal == True

    assert len(expected_outside_2_pi) == len(result_outside_2_pi)
    equal = True
    for i in range(len(expected_outside_2_pi)):
        if np.absolute(expected_outside_2_pi[i] - result_outside_2_pi[i]) > 0.01:
            equal = False
            break
    assert equal == True

def test_calculate_mean_orientations():
    orientations = []
    expected = []
    result = sorient.calculate_mean_orientations(orientations=orientations)
    assert len(expected) == len(result)

    orientations = [[[0.2, 0.3], [0.4, 0.5], [0, 0], [0.6, 0.7], [0.8, 0.9]],
                    [[0.1, 0.2], [-0.1, -0.2], [0.3, 0], [0, 0], [1, 1]],
                    [[0, 0], [1, 1], [0.5, 0.5], [-0.5, -0.5], [-1, -1]]]
    
    expected = [[0.64018, 0.76822],
                [0.79262, 0.60971],
                [0, 0]]
    result = sorient.calculate_mean_orientations(orientations=orientations)

    assert len(expected) == len(result)
    equal = True
    for i in range(len(expected)):
        for j in range(2):
            if np.absolute(expected[i][j] - result[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

def test_get_differences():
    domain_size = (50, 50)
    # the results of this method is the distance squared
    positions_empty = []
    positions_diagonal = np.array([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]) # keep in mind that the domain is toroidal
    positions_disordered = np.array([[0.3, 17], [28, 5], [15, 15], [15, 20], [50, 5], [0, 46]])

    expected_empty = []
    expected_diagonal = [[0, 200, 800, 800, 200, 0],
                         [200, 0, 200, 800, 800, 200],
                         [800, 200, 0, 200, 800, 800],
                         [800, 800, 200, 0, 200, 800],
                         [200, 800, 800, 200, 0, 200],
                         [0, 200, 800, 800, 200, 0]]
    expected_disordered = [[0, 641.29, 220.09, 225.09, 144.09, 441.09],
                           [641.29, 0, 269, 394, 484, 565],
                           [220.09, 269, 0, 25, 325, 586],
                           [225.09, 394, 25, 0, 450, 801],
                           [144.09, 484, 325, 450, 0, 81],
                           [441.09, 565, 586, 801, 81, 0]]

    result_empty = sorient.get_differences(array=positions_empty, domain_size=domain_size)
    result_diagonal = sorient.get_differences(array=positions_diagonal, domain_size=domain_size)
    result_disordered = sorient.get_differences(array=positions_disordered, domain_size=domain_size)

    assert len(expected_empty) == len(result_empty)
    assert len(expected_diagonal) == len(result_diagonal)
    equal = True
    for i in range(len(expected_diagonal)):
        for j in range(len(expected_diagonal[0])):
            if np.absolute(expected_diagonal[i][j] - result_diagonal[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

    assert len(expected_disordered) == len(result_disordered)
    equal = True
    for i in range(len(expected_disordered)):
        for j in range(len(expected_disordered[0])):
            if np.absolute(expected_disordered[i][j] - result_disordered[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

def test_get_angle_differences():
    angles_empty = []
    angles_quarters = np.array([0, np.pi, 1.5*np.pi, 2*np.pi])
    angles_mixed = np.array([0, np.pi, -np.pi, -0.5*np.pi, 0.75*np.pi, 2*np.pi])

    expected_empty = []
    expected_quarters = [[0, -np.pi, -1.5*np.pi, -2*np.pi],
                         [np.pi, 0, -0.5*np.pi, -np.pi],
                         [1.5*np.pi, 0.5*np.pi, 0, -0.5*np.pi],
                         [2*np.pi, np.pi, 0.5*np.pi, 0]]
    expected_quarters_abs = [[0, np.pi, 1.5*np.pi, 2*np.pi],
                         [np.pi, 0, 0.5*np.pi, np.pi],
                         [1.5*np.pi, 0.5*np.pi, 0, 0.5*np.pi],
                         [2*np.pi, np.pi, 0.5*np.pi, 0]]
    expected_mixed = [[0, -np.pi, np.pi, 0.5*np.pi, -0.75*np.pi, -2*np.pi],
                      [np.pi, 0, 2*np.pi, 1.5*np.pi, 0.25*np.pi, -np.pi],
                      [-np.pi, -2*np.pi, 0, -0.5*np.pi, -1.75*np.pi, -3*np.pi],
                      [-0.5*np.pi, -1.5*np.pi, 0.5*np.pi, 0, -1.25*np.pi, -2.5*np.pi],
                      [0.75*np.pi, -0.25*np.pi, 1.75*np.pi, 1.25*np.pi, 0, -1.25*np.pi],
                      [2*np.pi, np.pi, 3*np.pi, 2.5*np.pi, 1.25*np.pi, 0]]
    expected_mixed_abs = [[0, np.pi, np.pi, 0.5*np.pi, 0.75*np.pi, 2*np.pi],
                      [np.pi, 0, 2*np.pi, 1.5*np.pi, 0.25*np.pi, np.pi],
                      [np.pi, 2*np.pi, 0, 0.5*np.pi, 1.75*np.pi, 3*np.pi],
                      [0.5*np.pi, 1.5*np.pi, 0.5*np.pi, 0, 1.25*np.pi, 2.5*np.pi],
                      [0.75*np.pi, 0.25*np.pi, 1.75*np.pi, 1.25*np.pi, 0, 1.25*np.pi],
                      [2*np.pi, np.pi, 3*np.pi, 2.5*np.pi, 1.25*np.pi, 0]]

    result_empty = sorient.get_angle_differences(angles=angles_empty)
    result_quarters = sorient.get_angle_differences(angles=angles_quarters)
    result_quarters_abs = sorient.get_angle_differences(angles=angles_quarters, return_absolute=True)
    result_mixed = sorient.get_angle_differences(angles=angles_mixed)
    result_mixed_abs = sorient.get_angle_differences(angles=angles_mixed, return_absolute=True)

    assert len(expected_empty) == len(result_empty)
    assert len(expected_quarters) == len(result_quarters)
    equal = True
    for i in range(len(expected_quarters)):
        for j in range(len(expected_quarters[0])):
            if np.absolute(expected_quarters[i][j] - result_quarters[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

    assert len(expected_quarters_abs) == len(result_quarters_abs)
    equal = True
    for i in range(len(expected_quarters_abs)):
        for j in range(len(expected_quarters_abs[0])):
            if np.absolute(expected_quarters_abs[i][j] - result_quarters_abs[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

    assert len(expected_mixed) == len(result_mixed)
    equal = True
    for i in range(len(expected_mixed)):
        for j in range(len(expected_mixed[0])):
            if np.absolute(expected_mixed[i][j] - result_mixed[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

    assert len(expected_mixed_abs) == len(result_mixed_abs)
    equal = True
    for i in range(len(expected_mixed_abs)):
        for j in range(len(expected_mixed_abs[0])):
            if np.absolute(expected_mixed_abs[i][j] - result_mixed_abs[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

def test_get_differences_sqrt():
    domain_size = (50, 50)
    # the results of this method is the distance squared
    positions_empty = []
    positions_diagonal = np.array([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]) # keep in mind that the domain is toroidal
    positions_disordered = np.array([[0.3, 17], [28, 5], [15, 15], [15, 20], [50, 5], [0, 46]])

    expected_empty = []
    expected_diagonal = [[0, 14.1421, 28.2843, 28.2843, 14.1421, 0],
                         [14.1421, 0, 14.1421, 28.2843, 28.2843, 14.1421],
                         [28.2843, 14.1421, 0, 14.1421, 28.2843, 28.2843],
                         [28.2843, 28.2843, 14.1421, 0, 14.1421, 28.2843],
                         [14.1421, 28.2843, 28.2843, 14.1421, 0, 14.1421],
                         [0, 14.1421, 28.2843, 28.2843, 14.1421, 0]]
    expected_disordered = [[0, 25.3237, 14.8354, 15.0029, 12.0037, 21.0021],
                           [25.3237, 0, 16.4012, 19.8494, 22, 23.7697],
                           [14.8354, 16.4012, 0, 5, 18.0278, 24.2074],
                           [15.003, 19.8494, 5, 0, 21.2132, 28.3019],
                           [12.0037, 22, 18.0277, 21.2132, 0, 9],
                           [21.0021, 23.7697, 24.2074, 28.3019, 9, 0]]

    result_empty = sorient.get_differences_sqrt(array=positions_empty, domain_size=domain_size)
    result_diagonal = sorient.get_differences_sqrt(array=positions_diagonal, domain_size=domain_size)
    result_disordered = sorient.get_differences_sqrt(array=positions_disordered, domain_size=domain_size)

    assert len(expected_empty) == len(result_empty)
    assert len(expected_diagonal) == len(result_diagonal)
    equal = True
    for i in range(len(expected_diagonal)):
        for j in range(len(expected_diagonal[0])):
            if np.absolute(expected_diagonal[i][j] - result_diagonal[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

    assert len(expected_disordered) == len(result_disordered)
    equal = True
    for i in range(len(expected_disordered)):
        for j in range(len(expected_disordered[0])):
            if np.absolute(expected_disordered[i][j] - result_disordered[i][j]) > 0.01:
                equal = False
                break
    assert equal == True

def run_all():
    test_normalize_orientations()
    test_normalize_angles()
    test_calculate_mean_orientations()
    test_get_differences()
    test_get_angle_differences()
    test_get_differences_sqrt()
    print("Everything passed for service_orientations")