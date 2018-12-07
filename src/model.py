import numpy as np
from functions import eigen_order
from sklearn.cluster import KMeans


def pca(features, m_pca=None):
    rows, cols = features.shape

    mu = np.mean(features, axis=0)

    st = (features - mu[None, :]).T.dot(features - mu[None, :])

    if m_pca is None:
        m_pca = int(cols * 6 / 10)

    u = eigen_order(st, m=m_pca)

    return u, mu

