import numpy as np


# Partitions features related to training.
def set_feat_train(features, train_idx):
    feat_train = []

    for idx in train_idx:
        feat_train.append(features[idx])

    return feat_train


# Partitions features related to training and features related to validation.
def set_feat_train_valid(features, train_idx, n_clusters_valid, labels):
    feat_train, feat_valid = [], []

    clusters = np.random.permutation(np.unique(labels[train_idx]))

    clusters_valid = clusters[0:n_clusters_valid]

    valid_idx = np.array([], dtype=int)

    for idx in train_idx:
        if labels[idx] in clusters_valid:
            feat_valid.append(features[idx])
            valid_idx = np.append(valid_idx, idx, axis=None)
        else:
            feat_train.append(features[idx])

    return feat_train, feat_valid, valid_idx
