import numpy as np


def softmax(f):
    # to make the result numerical stale
    f -= np.max(f,axis=0)
    return np.exp(f)/np.sum(np.exp(f),axis=0)