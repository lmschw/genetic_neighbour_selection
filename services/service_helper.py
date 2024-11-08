from sklearn import preprocessing
import numpy as np

"""
Contains helper methods that do not fit the themes of the other services.
"""

def normalise(values, norm='l1'):
    """
    Applies the Euclidean norm to the values.

    Params:
        - values (np.array): the values to be normalised

    Returns:
        A numpy array with the normalised values.
    """
    if norm == 'l0':
        return np.count_nonzero(values)

    normalised_vector = preprocessing.normalize(X=[values], norm=norm)[0]
    return normalised_vector