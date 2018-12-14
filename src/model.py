import numpy as np
from functions import eigen_order


# Function for basic PCA.
def pca(features, m_pca=None):
    rows, cols = features.shape

    mu = np.mean(features, axis=0)

    st = (features - mu[None, :]).T.dot(features - mu[None, :])

    if m_pca is None:
        m_pca = int(cols * 6 / 10)

    u = eigen_order(st, m=m_pca)

    return u, mu


# Compute gradient of sum of distances between points of dissimilar classes. Returns g and dg
def compute_dg(features, labels, quad_u):
    rows, cols = features.shape

    g, dg = 0, np.zeros(cols)
    for i in range(rows-1):
        for j in range(i+1, rows):
            if labels[i] != labels[j]:
                y2 = np.square(features[i, :] - features[j, :])
                dist = np.sqrt(quad_u.dot(y2))
                g += dist
                dg += y2 / (2 * dist)

    return g, dg


# Compute sum of element-wise square of differences between points of similar classes.
def compute_sim_feat(features, labels):
    rows, cols = features.shape

    y2 = np.zeros(cols)
    for i in range(rows-1):
        for j in range(i+1, rows):
            if labels[i] == labels[j]:
                y2 += np.square(features[i, :] - features[j, :])
            else:
                break

    return y2


# Iterative Projection steps 1 & 2.
def iter_project(quad_u, y2, f, obj_f):

    lam = (f - obj_f) / (y2.dot(y2))
    quad_u_nxt = quad_u - y2 * lam

    idx = np.transpose(np.argwhere(quad_u_nxt < 0))[0]
    quad_u_nxt[idx] = 0

    return quad_u_nxt


# Gradient ascent algorithm with Iterative Projection.
def optimize_metric(features, labels, max_iter=20, alpha=1e-11, tol=1e-1, tol_f=1e-3, obj_f=1):
    cols = features.shape[1]

    quad_u = np.ones(cols) / cols

    y2 = compute_sim_feat(features, labels)
    eps, n_iter, g = 1, 0, 0.0
    while eps > tol and n_iter < max_iter:
        f = quad_u.dot(y2)
        print('g = %.2f' % g, '/ f = %.2f' % f)
        print('Projecting...')
        while np.abs(f - obj_f) > tol_f:
            quad_u = iter_project(quad_u, y2, f, obj_f)
            f = quad_u.dot(y2)

        print('Ascending...')
        g, dg = compute_dg(features, labels, quad_u)

        quad_u_nxt = quad_u + alpha * dg
        eps = np.linalg.norm(quad_u_nxt - quad_u) / np.linalg.norm(quad_u)
        quad_u = quad_u_nxt

        n_iter += 1

    g_mat = np.sqrt(np.diag(quad_u))

    return g_mat, n_iter
