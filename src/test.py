import numpy as np
from functions import kmetric, knn, result_display


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
def rank_query(features, query_idx, gallery_idx, file_list, labels, cam_idx, quad_mat=None, rank=1, display=False):
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

        if quad_mat is None:
            k_idx = knn(np.array(features[idx]), np.array(feat_gall_cam_rem), k=rank)
        else:
            k_idx = kmetric(np.array(features[idx]), np.array(feat_gall_cam_rem), quad_mat, k=rank)

        gallery_id = labels[gall_cam_rem_idx[k_idx]]
        file_idx = np.concatenate((idx, gall_cam_rem_idx[k_idx]), axis=None)

        for j in range(rank):
            if query_id == gallery_id[j]:
                color[j+1] = 1
                tp += 1
                rank_score[i] = 1
            else:
                color[j+1] = 2
                fp += 1

        print('-- Query:', query_id, '/ Gallery:', gallery_id, '/ Retrieval:', rank_score[i])
        if display:
            result_display(rank, color, file_list[file_idx])

        i += 1

    rank_score = np.mean(rank_score, axis=None)
    ma_prec = tp / (tp + fp)

    return rank_score, ma_prec



