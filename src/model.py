import numpy as np
from functions import eigen_order


def lda(features, labels, m_lda=None, n_clusters=1467):
    rows, cols = features.shape

    mu = np.mean(features, axis=0)

    sw = np.zeros((cols, cols))
    mu_cluster = np.zeros((n_clusters, cols))
    for k in range(0, n_clusters):
        k_set = np.transpose(np.argwhere(labels == k))[0]

        if k_set.size == 0:
            mu_cluster[k, :] = np.zeros(cols)
        else:
            mu_cluster[k, :] = np.mean(features[k_set, :], axis=0)

        if k_set.size > 1:
            mu_w = mu_cluster[k, :]
            sw = sw + (features[k_set, :] - mu_w[None, :]).T.dot((features[k_set, :] - mu_w[None, :]))

    sb = (mu_cluster - mu[None, :]).T.dot((mu_cluster - mu[None, :]))

    if m_lda is None:
        m_lda = int(cols * 2 / 3)

    w = eigen_order(np.linalg.inv(sw).dot(sb), m=m_lda)

    return w, mu


class randSmpFeatSubmod:
    def __init__(self, model_id, features, labels, n_clusters, m0, m1):
        self.model_id = model_id
        self.features = features
        self.labels = labels
        self.n_clusters = n_clusters
        self.m0 = m0
        self.m1 = m1
        self.data_train_proj = None
        self.w = None
        self.mu = None

    def setup(self):
        print('Building Random Feature Sampling sub-model', self.model_id, '...')
        array = np.random.permutation(np.arange(self.m0, self.features.shape[1] - self.n_clusters))
        m_ar = np.concatenate((np.arange(self.m0), array[0:self.m1]), axis=None)
        self.data_train_proj, self.w, self.mu = lda(self.features, self.labels,
                                                    m_lda=m_ar, n_clusters=self.n_clusters)
        print('sub-model', self.model_id, 'done!')
