import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Returns indices of k features closest to input feature according to given metric.
def kmetric(x_s, x_data, metric=1, k=1):
    rows, cols = x_data.shape

    # metric: {'l1': 0, 'l2': 1, 'cos': 2, 'maha': 3, 'cheby': 4}
    if metric == 0:
        x_dist = np.sum(np.abs(x_data - x_s[None, :]), axis=1)
    elif metric == 2:
        numerator = x_s.dot(x_data.T)
        denominator = np.linalg.norm(x_data, axis=1) * np.linalg.norm(x_s[None, :], axis=1)
        x_dist = np.ones(rows) - numerator / denominator
    elif metric == 3:
        s_inv = np.linalg.inv(np.cov(x_data, ddof=0, rowvar=False))
        x_dist = np.sqrt(np.diag((x_data - x_s[None, :]).dot(s_inv.dot((x_data - x_s[None, :]).T))))
    elif metric == 4:
        x_dist = np.max(np.abs(x_data - x_s[None, :]), axis=1)
    else:
        x_dist = np.linalg.norm(x_data - x_s[None, :], axis=1)

    idx = np.argsort(x_dist)[0:k]

    return idx


def eigen_order(s, m=None):
    lc, vc = np.linalg.eig(s)

    indices = np.argsort(np.abs(lc))[::-1]

    v = np.abs(vc[:, indices])[:, 0:m]

    return v


def conf_mat(y_actu, y_pred):
    cm = confusion_matrix(y_actu, y_pred)

    plt.figure()
    plt.matshow(cm, cmap='Blues')
    plt.colorbar()
    plt.ylabel('Actual id')
    plt.xlabel('Predicted id')
    plt.show()
