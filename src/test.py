import numpy as np
from functions import rank_display, knn


def set_feat_query_gallery(features, query_idx, gallery_idx):
    feat_query, feat_gallery = [], []

    for idx in query_idx:
        feat_query.append(features[idx - 1])

    for idx in gallery_idx:
        feat_gallery.append(features[idx - 1])

    return feat_query, feat_gallery


def rank_query(features, query_idx, gallery_idx, file_list, labels, clusters_means, rank=1):
    feat_query, feat_gallery = set_feat_query_gallery(features, query_idx, gallery_idx)

    n_query = query_idx.size

    color = np.zeros((n_query, rank+1), dtype=int)

    for i in range(n_query):
        knn_idx = knn(np.array(feat_query[i]), np.array(feat_gallery), k=rank)

        kmeans_idx = np.zeros(rank, dtype=int)
        for j in range(rank):
            kmeans_idx[j] = knn(np.array(feat_gallery[knn_idx[j]]), clusters_means)

        id_query = labels[query_idx[i] - 1]
        id_gallery = labels[gallery_idx[knn_idx] - np.ones(rank, dtype=int)]
        id_lloyds = kmeans_idx

        for j in range(rank):
            if id_query == id_pred[j]:
                color[i, j+1] = 1
            else:
                color[i, j+1] = 2

    return color
