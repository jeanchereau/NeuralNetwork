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


def compute_g(features, labels, quad_mat):
    rows, cols = features.shape

    g, dg = 0, np.zeros((cols, cols))
    for i in range(0, rows-1):
        for j in range(i+1, rows):
            if labels[i] != labels[j]:
                y = features[i, :] - features[j, :]
                dist = np.sqrt(y.dot(quad_mat.dot(y.T)))
                g += dist
                dg += y.T.dot(y) / (2 * dist)

    return g, dg


def compute_f(features, labels, quad_mat):
    rows, cols = features.shape

    f, df = 0, np.zeros((cols, cols))
    for i in range(0, rows-1):
        for j in range(i+1, rows):
            if labels[i] == labels[j]:
                y = features[i, :] - features[j, :]
                f += y.dot(quad_mat.dot(y.T))
                df += y.T.dot(y)
            else:
                break

    return f, df


def optimize_metric(features, labels, max_iter=300):
    rows, cols = features.shape

    quad_mat = np.random.rand(cols, cols)
    quad_mat = quad_mat.dot(quad_mat.T)
    quad_mat = quad_mat / np.trace(quad_mat)

    quad_mat_nxt = quad_mat + 1
    quad_mat_pre = None
    dg_pre = None

    tol = 0.1
    eps = np.linalg.norm(quad_mat_nxt - quad_mat, ord='fro')

    n_iter = 0
    while eps > tol and n_iter < max_iter:
        print(quad_mat)

        f, df = compute_f(features, labels, quad_mat)
        while f > 230:
            print(f)
            quad_mat = quad_mat / np.trace(quad_mat)
            lc, vc = np.linalg.eig(quad_mat)
            l, v = lc.real, vc.real
            l_idx = np.argwhere(l < 0)
            l[l_idx] = 0
            quad_mat = v.T.dot(np.diag(l).dot(v))
            
            f, df = compute_f(features, labels, quad_mat)

        print('yo')

        g, dg = compute_g(features, labels, quad_mat)

        print('hi')

        if quad_mat_pre is None or dg_pre is None:
            alpha = 1
            quad_mat_nxt = quad_mat + alpha * dg
        else:
            alpha = (quad_mat - quad_mat_pre).T.dot(dg - dg_pre) / eps
            quad_mat_nxt = quad_mat + alpha.dot(dg)

        dg_pre = dg
        eps = np.linalg.norm(quad_mat_nxt - quad_mat, ord='fro')
        quad_mat_pre = quad_mat
        quad_mat = quad_mat_nxt

        n_iter += 1
        print(n_iter)

    g_mat = np.linalg.cholesky(quad_mat)

    return g_mat, n_iter
