import numpy as np
from functions import eigen_order


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

    clusters = np.unique(labels).size

    g, dg = 0, np.zeros((cols, cols))
    for i in range(0, rows-1):
        for j in range(i+1, rows):
            if labels[i] != labels[j]:
                y = (features[i, :] - features[j, :]) / np.sqrt(clusters)
                dist = np.asscalar(np.sqrt(y[None, :].dot(quad_mat.dot(y[:, None]))))
                g += dist
                dg += y[:, None].dot(y[None, :]) / (2 * dist)

    return g, dg


def compute_f(features, labels, quad_mat):
    rows, cols = features.shape

    clusters = np.unique(labels).size

    f, df, y = 0, np.zeros((cols, cols)), np.zeros(cols)
    for i in range(0, rows-1):
        for j in range(i+1, rows):
            if labels[i] == labels[j]:
                y += features[i, :] - features[j, :]
            else:
                break

    y = y / np.sqrt(clusters)
    f = np.asscalar(y[None, :].dot(quad_mat.dot(y[:, None])))
    df = y[:, None].dot(y[None, :])
    
    return f, df, y


def qp_project(quad_mat, f, df, y, obj_f):

    lam = (f - obj_f) / (y.dot(y) * y.dot(y))
    quad_mat_nxt = quad_mat - df * lam

    lc, vc = np.linalg.eig(quad_mat_nxt)
    l, v = lc.real, vc.real
    l_idx = np.transpose(np.argwhere(l < 0))[0]
    l[l_idx] = 0
    quad_mat_nxt = v.T.dot(np.diag(l).dot(v))

    return quad_mat_nxt


def optimize_metric(features, labels, max_iter=10, n_part=1, alpha=1e-14, tol=1e-3, tol_f=1e-1, obj_f=1):
    cols = features.shape[1]

    quad_mat = np.zeros((cols, cols))

    for i in range(0, n_part):
        # feat_tmp = features[1, :]

        quad_mat_tmp = np.random.randn(cols, cols)
        quad_mat_tmp = quad_mat_tmp.dot(quad_mat_tmp.T)
        quad_mat_tmp = quad_mat_tmp / np.trace(quad_mat_tmp)

        eps, n_iter, g = 1, 0, 0.0
        while eps > tol and n_iter < max_iter:

            f, df, y = compute_f(features, labels, quad_mat_tmp)
            while np.abs(f - obj_f) > tol_f:
                print('Projecting...')
                quad_mat_tmp = qp_project(quad_mat_tmp, f, df, y, obj_f)
                f, df, y = compute_f(features, labels, quad_mat_tmp)
                print('g = %.2f' % g, '/ f = %.2f' % f, '/ difference = %.5f' % eps)

            print('Ascending...')
            g, dg = compute_g(features, labels, quad_mat_tmp)
            quad_mat_tmp_nxt = quad_mat_tmp + alpha * dg

            eps = np.linalg.norm(quad_mat_tmp_nxt - quad_mat_tmp, ord='fro') / np.linalg.norm(quad_mat_tmp, ord='fro')

            quad_mat_tmp = quad_mat_tmp_nxt

            n_iter += 1

        quad_mat += quad_mat_tmp

    quad_mat = quad_mat / n_part

    g_mat = np.linalg.cholesky(quad_mat)

    return g_mat, n_iter
