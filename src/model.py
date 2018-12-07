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


def optimize_metric(features, max_iter=300):
    rows, cols = features.shape
    quad_mat = np.zeros((cols, cols))
    n_iter = 0

    while n_iter < max_iter:
        n_iter += 1

    g_mat = np.linalg.cholesky(quad_mat)

    return g_mat, n_iter
