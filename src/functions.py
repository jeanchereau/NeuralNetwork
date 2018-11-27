import numpy as np


# Returns indices of k features closest to input feature.
def knn(x_s, x_data, k=1):
    cols = x_s.size

    x_dist = np.linalg.norm(x_data - x_s.reshape(1, cols), axis=1)

    idx = np.argsort(x_dist)[0:k]

    return idx


# Sort the indices of the cluster means.
def cluster_means_sort(cluster_means, labels):
    n_clusters, n_features = cluster_means.shape

    cluster_means_sorted = np.zeros((n_clusters, n_features))

    labels_corr = [labels[0]]

    for i in range(1, labels.size):
        if labels[i] != labels[i-1] and not labels[i] in labels_corr:
            labels_corr.append(labels[i])

    print(len(labels_corr), n_clusters)

    for i in range(0, n_clusters):
        cluster_means_sorted[i] = cluster_means[labels_corr[i]]

    return cluster_means_sorted
