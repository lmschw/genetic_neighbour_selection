import numpy as np

import services.service_helper as shelp

def test_normalise_l0():
    vals_empty = np.array([])
    vals_zeros = np.zeros(30)
    vals_pos_full = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    vals_mixed_full = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
    vals_pos_with_zeros = np.array([1, 0, 2, 3, 4, 0, 0, 0, 5, 0])
    vals_mixed_with_zeros = np.array([0, -1, 2, 0, 3, -3, 0, 0, 5, -5])

    norm_empty = shelp.normalise(values=vals_empty, norm='l0')
    norm_zeros = shelp.normalise(values=vals_zeros, norm='l0')
    norm_pos_full = shelp.normalise(values=vals_pos_full, norm='l0')
    norm_mixed_full = shelp.normalise(values=vals_mixed_full, norm='l0')
    norm_pos_with_zeros = shelp.normalise(values=vals_pos_with_zeros, norm='l0')
    norm_mixed_with_zeros = shelp.normalise(values=vals_mixed_with_zeros, norm='l0')

    assert norm_empty == 0, "test_normalise_l0: When no non-zero value is present, the result should be 0"
    assert norm_zeros == 0, "test_normalise_l0: 30 zero values should result in 0"
    assert norm_pos_full == 10, "test_normalise_l0: 10 positive values should result in 10"
    assert norm_mixed_full == 10, "test_normalise_l0: 10 non-zero values should result in 10"
    assert norm_pos_with_zeros == 5, "test_normalise_l0: 5 positive values should result in 5"
    assert norm_mixed_with_zeros == 6, "test_normalise_l0: 6 positive values should result in 6"


def test_normalise_l1():
    vals_empty = np.array([])
    vals_zeros = np.zeros(30)
    vals_pos = np.random.uniform(low=0, high=5, size=30)
    vals_neg_pos = np.random.uniform(low=-5, high=5, size=30)

    norm_empty = shelp.normalise(values=vals_empty, norm='l1')
    norm_zeros = shelp.normalise(values=vals_zeros, norm='l1')
    norm_pos = shelp.normalise(values=vals_pos, norm='l1')
    norm_neg_pos = shelp.normalise(values=vals_neg_pos, norm='l1')

    assert len(norm_empty) == 0, "test_normalise_l1: An empty array cannot be normalised and should therefore not change"
    assert sum(norm_zeros) == 0, "test_normalise_l1: Sum should be zero for zero values"
    assert sum(norm_pos) >= 0.999, "test_normalise_l1: Sum should be close to one for positive values"
    assert sum(np.absolute(norm_neg_pos)) >= 0.999, "test_normalise_l1: Sum should be close to one for mixed values"

def test_normalise_2d_l0():
    vals_2d_zeros = np.zeros((5, 10))
    vals_2d_pos = np.array([[1, 0, 3, 4, 0, 0, 7, 0, 9, 0],
                            [0, 3, 0, 0, 0, 7, 8, 9, 0, 11],
                            [3, 4, 0, 6, 0, 8, 0, 10, 0, 0],
                            [0, 0, 6, 0, 8, 9, 0, 11, 0, 13],
                            [5, 6, 0, 0, 0, 10, 0, 12, 0, 14]])
    vals_2d_neg_pos = np.array([[1, 0, 3, -4, 0, 0, 7, -8, 0, 0],
                                [0, 3, -4, 0, 0, 7, -8, 0, -10, 0],
                                [3, 0, 0, -6, 0, 0, 9, -10, 0, -12],
                                [-4, 0, -6, 0, 0, 9, -10, 0, 0, 13],
                                [0, -6, 0, -8, 9, -10, 11, 0, 0, 0]])
    
    norm_2d_zeros = shelp.normalise(values=vals_2d_zeros, norm='l0')
    norm_2d_pos = shelp.normalise(values=vals_2d_pos, norm='l0')
    norm_2d_neg_pos = shelp.normalise(values=vals_2d_neg_pos, norm='l0')

    for row in range(5):
        assert norm_2d_zeros[row] == 0, "test_normalise_l0: L0 should be zero for all zeros"
        assert norm_2d_pos[row] ==5, "test_normalise_2d_l0: L0 should be 5 for 5 positive non-zero values"
        assert norm_2d_neg_pos[row] ==5, "test_normalise_2d_l0: L0 should be 5 for 5 non-zero values"

def test_normalise_2d_l1():
    vals_2d_zeros = np.zeros((5, 10))
    vals_2d_pos = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
    vals_2d_neg_pos = np.array([[1, -2, 3, -4, 5, -6, 7, -8, 9, -10],
                                [-2, 3, -4, 5, -6, 7, -8, 9, -10, 11],
                                [3, -4, 5, -6, 7, -8, 9, -10, 11, -12],
                                [-4, 5, -6, 7, -8, 9, -10, 11, -12, 13],
                                [5, -6, 7, -8, 9, -10, 11, -12, 13, -14]])
    
    norm_2d_zeros = shelp.normalise(values=vals_2d_zeros, norm='l1')
    norm_2d_pos = shelp.normalise(values=vals_2d_pos, norm='l1')
    norm_2d_neg_pos = shelp.normalise(values=vals_2d_neg_pos, norm='l1')

    for row in range(5):
        assert sum(norm_2d_zeros[row]) == 0, "test_normalise_l1: Sum should be zero for zero values"
        assert sum(norm_2d_pos[row]) >= 0.999, "test_normalise_2d_l1: Sum should be close to one for positive values"
        assert sum(np.absolute(norm_2d_neg_pos[row])) >= 0.999, "test_normalise_2d_l1: Sum should be close to one for mixed values"

def test_normalise():
    # 1d array
    test_normalise_l0()
    test_normalise_l1()
    # 2d array
    test_normalise_2d_l0()
    test_normalise_2d_l1()

def test_get_neighbours():
    domain_size = (50, 50)
    # no neighbours
    positions = np.array([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]])
    expected = np.array([[True, False, False, False, False],
                         [False, True, False, False, False],
                         [False, False, True, False, False],
                         [False, False, False, True, False],
                         [False, False, False, False, True]])
    result = shelp.get_neighbours(positions=positions, domain_size=domain_size, radius=5)
    equal = True
    for i in range(len(positions)):
        for j in range(len(positions)):
            if result[i][j] != expected[i][j]:
                equal = False
                break
    assert equal == True

    # single neighbour
    positions = np.array([[0, 0], [2, 2], [20, 20], [22, 22]])
    expected = np.array([[True, True, False, False],
                         [True, True, False, False],
                         [False, False, True, True],
                         [False, False, True, True],])
    result = shelp.get_neighbours(positions=positions, domain_size=domain_size, radius=5)
    equal = True
    for i in range(len(positions)):
        for j in range(len(positions)):
            if result[i][j] != expected[i][j]:
                equal = False
                break
    assert equal == True

    # mixed
    positions = np.array([[0, 0], [2, 2], [20, 20], [22, 22], [25, 24], [30, 30], [48, 48], [49, 49]])
    expected = np.array([[True, True, False, False, False, False, True, True],
                         [True, True, False, False, False, False, False, True],
                         [False, False, True, True, False, False, False, False],
                         [False, False, True, True, True, False, False, False],
                         [False, False, False, True, True, False, False, False],
                         [False, False, False, False, False, True, False, False],
                         [True, False, False, False, False, False, True, True],
                         [True, True, False, False, False, False, True, True]])
    result = shelp.get_neighbours(positions=positions, domain_size=domain_size, radius=5)
    equal = True
    for i in range(len(positions)):
        for j in range(len(positions)):
            if result[i][j] != expected[i][j]:
                equal = False
                break
    assert equal == True

    # all neighbours
    positions = np.array([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]])
    expected = np.full((5,5), True)
    result = shelp.get_neighbours(positions=positions, domain_size=domain_size, radius=50)
    equal = True
    for i in range(len(positions)):
        for j in range(len(positions)):
            if result[i][j] != expected[i][j]:
                equal = False
                break
    assert equal == True


def run_all():
    test_normalise()
    test_get_neighbours()
    print("Everything passed for service_helper")
