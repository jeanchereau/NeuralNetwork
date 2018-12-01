import numpy as np


# Returns indices of k features closest to input feature according to given metric.
def kmetric(x_s, x_data, metric=1, k=1):
    rows, cols = x_data.shape

    # metric: ['l1', 'l2', 'cos', 'maha', 'chi']

    if metric == 0:
        x_dist = np.sum(np.abs(x_data - x_s.reshape(1, cols)), axis=1)
    elif metric == 2:
        numerator = x_s.reshape(1, cols).dot(x_data.T)
        denominator = np.linalg.norm(x_data, axis=1) * np.linalg.norm(x_s.reshape(1, cols), axis=1)

        x_dist = np.ones((1, rows)) - numerator / denominator
    elif metric == 3:
        s_inv = np.linalg.inv(np.cov(x_data, ddof=0, rowvar=True))

        x_dist = np.sqrt((x_data - x_s.reshape(1, cols)).T.dot(s_inv.dot((x_data - x_s.reshape(1, cols)))))
    elif metric == 4:
        numerator = np.square(x_data - x_s.reshape(1, cols))
        denominator = x_data + x_s.reshape(1, cols)

        x_dist = np.sum(numerator / denominator, axis=1) / 2
    else:
        x_dist = np.linalg.norm(x_data - x_s.reshape(1, cols), axis=1)

    idx = np.argsort(x_dist)[0:k]

    return idx
