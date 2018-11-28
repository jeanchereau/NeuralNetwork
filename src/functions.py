import numpy as np


# Returns indices of k features closest to input feature.
def knn(x_s, x_data, k=1):
    cols = x_s.size

    x_dist = np.linalg.norm(x_data - x_s.reshape(1, cols), axis=1)

    idx = np.argsort(x_dist)[0:k]

    return idx
