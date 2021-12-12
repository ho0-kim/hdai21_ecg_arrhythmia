import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample

def minmixscale(X):
    minmax = MinMaxScaler()
    for i in range(len(X)):
        X[i] = minmax.fit_transform(X[i])
    return X

def elevation(X, n_leads=2):
    X = X * 1e6
    for _x in X:
        for i in range(n_leads):
            counts = np.bincount(_x[:,i].astype(int))
            _x[:,i] = _x[:,i] - np.argmax(counts)
    X = X * 1e-6

def stretch(x, n_leads=2):
    l = int(5000 * (1 + (np.random.random()-0.5)/3))
    y = resample(x, l)
    
    if l < 5000:
        y_ = np.zeros(shape=(5000,n_leads))
        
        y_[:l] = y
    else:
        y_ = y[:5000]
    return y_

def amplify(x):
    alpha = (np.random.random()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor

def augment(x, n_leads=2):
    if np.random.random() < 0.33:
        new_y = stretch(x, n_leads)
    elif np.random.random() < 0.66:
        new_y = amplify(x)
    else:
        new_y = stretch(x, n_leads)
        new_y = amplify(new_y)
    return new_y

def add_gaussian_noise(X, n_leads=2):
    for _x in X:
        _x = _x + np.random.normal(0,.5,(5000,n_leads))
    return X