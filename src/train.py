import numpy as np


def set_feat_train(features, train_idx):
    feat_train = []

    for idx in train_idx:
        feat_train.append(features[idx - 1])

    return feat_train


def set_feat_train_valid(features, train_idx, n_clusters, n_clusters_valid, labels):
    feat_train, feat_valid = [], []

    clusters = np.random.permutation(np.arange(1, n_clusters+1))

    # clusters_valid, clusters_train = clusters[0:n_clusters_valid], clusters[n_clusters_valid:None]

    sub_train_idx, valid_idx = np.array([]), np.array([])

    for i in range(0, n_clusters):
        cluster = clusters[i]

        idx_set = np.transpose(np.argwhere(labels == cluster))[0]
        idx_set = idx_set + np.ones(idx_set.size, dtype=int)

        if i < n_clusters_valid:

            valid_idx = np.concatenate((valid_idx, train_idx[]), axis=None)

        else:
            sub_train_idx = np.concatenate((sub_train_idx, train_idx[]), axis=None)

    for idx in train_idx:
        if idx in sub_train_idx:
            feat_train.append(features[idx - 1])
        elif idx in valid_idx:
            feat_valid.append(features[idx - 1])

    return feat_train, feat_valid