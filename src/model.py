import numpy as np
from functions import eigen_order


def pca_lda(features, labels, m_lda=None, m_pca=None, n_clusters=1467):
    rows, cols = features.shape

    mu = np.mean(features, axis=2)

    sw = np.zeros((cols, cols))
    mu_cluster = np.zeros((n_clusters, cols))
    for k in range(0, n_clusters):
        k_set = np.transpose(np.argwhere(labels == k))[0]

        if k_set.size == 0:
            mu_cluster[:, k] = np.zeros(cols)
        else:
            mu_cluster[:, k] = np.mean(features[:, k_set], axis=1)

        if k_set.size > 1:
            mu_w = mu_cluster[:, k]
            sw = sw + (features[:, k_set] - mu_w[:, None]).dot((features[:, k_set] - mu_w[:, None]).T)

    sb = (mu_cluster - mu[:, None]).dot((mu_cluster - mu[:, None]).T)

    if m_pca is None:
        m_pca = n_clusters - 1
        if m_lda is None:
            m_lda = int(m_pca * 2 / 3)
    elif m_lda is None and type(m_pca) is not np.ndarray:
        if m_pca > n_clusters - 1:
            m_lda = n_clusters - 1
        else:
            m_lda = m_pca

    if type(m_pca) is np.ndarray:
        i_set = np.transpose(np.argwhere(m_pca > cols - 1 - n_clusters))[0]
        m_pca[i_set] = (cols - 1 - n_clusters) * np.ones(i_set.size)
        if m_lda is None:
            if m_pca.size > n_clusters - 1:
                m_lda = n_clusters - 1
            else:
                m_lda = m_pca.size

    st = (features - mu[:, None]).T.dot(features - mu[:, None])
    u = (features - mu[:, None]).dot(eigen_order(st, m=m_pca))

    slda = np.linalg.inv(u.T.dot(sw.dot(u))).dot(u.T.dot(sb.dot(u)))
    w = u.dot(eigen_order(slda, m=m_lda))

    features_proj = w.T.dot(features - mu[:, None])

    return features_proj


class randSmpFeatSubmod:
    def __init__(self, model_id, data_train, data_id_memory, n_p, m0, m1):
        self.model_id = model_id
        self.data_train = data_train
        self.data_id_memory = data_id_memory
        self.n_p = n_p
        self.m0 = m0
        self.m1 = m1
        self.data_train_proj = None
        self.w = None
        self.mu = None

    def setup(self):
        print('Building Random Feature Sampling sub-model', self.model_id, '...')
        array = np.random.permutation(np.arange(self.m0, self.data_train.shape[1] - self.n_p))
        m_ar = np.concatenate((np.arange(self.m0), array[0:self.m1]), axis=None)
        self.data_train_proj, self.w, self.mu = pca_lda(self.data_train, self.data_id_memory, m_pca=m_ar, n_p=self.n_p)
        print('sub-model', self.model_id, 'done!')
