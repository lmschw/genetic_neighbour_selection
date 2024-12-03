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

def run_all():
    print("Tests for service_helper starting")
    print("Tests for method 'normalise()' starting")
    test_normalise_l0()
    test_normalise_l1()
    print("Everything passed")
