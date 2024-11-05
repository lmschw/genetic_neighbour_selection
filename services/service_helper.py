from sklearn import preprocessing

def normalise(values):
    normalised_vector = preprocessing.normalize(X=[values], norm='l2')[0]
    return values/normalised_vector