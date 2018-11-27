import numpy as np
import matplotlib.pyplot as plt
from functions import knn


def set_feat_query_gallery(features, query_idx, gallery_idx):
    feat_query, feat_gallery = [], []

    for idx in query_idx:
        feat_query.append(features[idx - 1])

    for idx in gallery_idx:
        feat_gallery.append(features[idx - 1])

    return feat_query, feat_gallery


def set_feat_cam1_cam2(feat_gallery, gallery_idx, cam_id):
    feat_gall_cam1, feat_gall_cam2 = [], []
    gall_cam1_idx, gall_cam2_idx = np.array([], dtype=int), np.array([], dtype=int)

    i = 0
    for idx in gallery_idx:
        if cam_id[idx - 1] == 1:
            feat_gall_cam1.append(feat_gallery[i])
            gall_cam1_idx = np.append(gall_cam1_idx, idx, axis=None)
        else:
            feat_gall_cam2.append(feat_gallery[i])
            gall_cam2_idx = np.append(gall_cam2_idx, idx, axis=None)

        i += 1

    return feat_gall_cam1, gall_cam1_idx, feat_gall_cam2, gall_cam2_idx


def rank_query(features, query_idx, gallery_idx, file_list, labels, clusters_means, cam_id, rank=1):
    feat_query, feat_gallery = set_feat_query_gallery(features, query_idx, gallery_idx)

    feat_gall_cam1, gall_cam1_idx, feat_gall_cam2, gall_cam2_idx = set_feat_cam1_cam2(feat_gallery, gallery_idx, cam_id)

    n_query = query_idx.size

    color = np.zeros(rank+1, dtype=int)

    for i in range(n_query):
        cluster_idx = labels[query_idx[i] - 1] - 1

        if cam_id[query_idx[i] - 1] == 1:
            k_idx = knn(clusters_means[cluster_idx, :], np.array(feat_gall_cam2), k=rank)
            id_gallery = labels[gall_cam2_idx[k_idx] - np.ones(rank, dtype=int)]
        else:
            k_idx = knn(clusters_means[cluster_idx, :], np.array(feat_gall_cam1), k=rank)
            id_gallery = labels[gall_cam1_idx[k_idx] - np.ones(rank, dtype=int)]

        id_query = labels[query_idx[i] - 1]

        for j in range(rank):
            if id_query == id_gallery[j]:
                color[j+1] = 1
            else:
                color[j+1] = 2

        # map = rank_map()
        # print(query_idx[i], map)
        file_idx = np.concatenate((query_idx[i], gallery_idx[k_idx]), axis=None)
        rank_display(rank, color, file_list[file_idx - np.ones(rank+1, dtype=int)])

    return color


def rank_display(rank, color, image_files):
    color_dict = ['black', 'green', 'red']
    w = 10
    h = 5
    fig = plt.figure(figsize=(h, w))
    cols = rank + 1
    rows = 1

    i = 0
    for image_file in image_files:
        image = plt.imread('../pr_data/images_cuhk03/' + image_file)
        fig.add_subplot(rows, cols, i+1)
        i += 1
        imgplot = plt.imshow(image)
        plt.setp(imgplot.spines.values(), color=color_dict[color[i]])

    plt.show()


def rank_map():
    pass
