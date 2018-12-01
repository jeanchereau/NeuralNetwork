import numpy as np
import matplotlib.pyplot as plt
from functions import kmetric


# Partitions features related to query and features related to gallery.
def set_feat_query_gallery(features, query_idx, gallery_idx):
    feat_query, feat_gallery = [], []

    for idx in query_idx:
        feat_query.append(features[idx])

    for idx in gallery_idx:
        feat_gallery.append(features[idx])

    return feat_query, feat_gallery


# Partitions features related to camera 1 gallery and features related to camera 2 gallery.
def rem_feat_cam_label(feat_gallery, gallery_idx, query_id, cam_id, labels, cam_idx):
    feat_gall_cam_rem = []
    gall_cam_rem_idx = np.array([], dtype=int)

    i = 0
    for idx in gallery_idx:
        if not (labels[idx] == query_id and cam_idx[idx] == cam_id):
            feat_gall_cam_rem.append(feat_gallery[i])
            gall_cam_rem_idx = np.append(gall_cam_rem_idx, idx, axis=None)

        i += 1

    return feat_gall_cam_rem, gall_cam_rem_idx


#
def rank_query(features, query_idx, gallery_idx, file_list, labels, cam_idx, cluster_means=None,
               metric=1, rank=1, display=False):
    feat_query, feat_gallery = set_feat_query_gallery(features, query_idx, gallery_idx)

    rank_score = np.zeros(len(feat_query))

    color = np.zeros(rank+1, dtype=int)
    i = 0
    tp, fp = 0, 0
    for idx in query_idx:
        query_id = labels[idx]
        cam_id = cam_idx[idx]

        feat_gall_cam_rem, gall_cam_rem_idx = rem_feat_cam_label(feat_gallery, gallery_idx, query_id, cam_id,
                                                                 labels, cam_idx)
        if cluster_means is not None:
            cluster_idx = kmetric(np.array(features[idx]), cluster_means)
            k_idx = kmetric(cluster_means[cluster_idx, :], np.array(feat_gall_cam_rem), metric=metric, k=rank)
        else:
            k_idx = kmetric(np.array(features[idx]), np.array(feat_gall_cam_rem), metric=metric, k=rank)

        gallery_id = labels[gall_cam_rem_idx[k_idx]]
        file_idx = np.concatenate((idx, gall_cam_rem_idx[k_idx]), axis=None)

        score = 0
        for j in range(rank):
            if query_id == gallery_id[j]:
                color[j+1] = 1
                tp += 1
                score = 1
            else:
                color[j+1] = 2
                fp += 1

        rank_score[i] = score
        i += 1

        if display:
            rank_display(rank, color, file_list[file_idx])

    rank_score = np.mean(rank_score, axis=None)
    ma_prec = tp / (tp + fp)

    return rank_score, ma_prec


def rank_display(rank, color, image_files):
    color_dict = ['black', 'green', 'red']
    w = rank + 3
    h = 3
    fig = plt.figure(figsize=(w, h))
    cols = rank + 1
    rows = 1

    i = 0
    for image_file in image_files:
        image = plt.imread('../pr_data/images_cuhk03/' + image_file[0])
        ax = fig.add_subplot(rows, cols, i+1)
        plt.imshow(image, aspect='auto')
        plt.setp(ax.spines.values(), color=color_dict[color[i]], linewidth=3)
        ax.tick_params(bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        i += 1

    plt.show()
