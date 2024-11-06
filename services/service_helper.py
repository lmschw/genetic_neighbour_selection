from sklearn import preprocessing

"""
Contains helper methods that do not fit the themes of the other services.
"""

def normalise(values):
    """
    Applies the Euclidean norm to the values.

    Params:
        - values (np.array): the values to be normalised

    Returns:
        A numpy array with the normalised values.
    """
    normalised_vector = preprocessing.normalize(X=[values], norm='l2')[0]
    return normalised_vector