from src.functions import rank_display


def set_feat_query_gallery(features, query_idx, gallery_idx):
    feat_query, feat_gallery = [], []

    for idx in query_idx:
        feat_query.append(features[idx - 1])

    for idx in gallery_idx:
        feat_gallery.append(features[idx - 1])

    return feat_query, feat_gallery


def rank_query(feat_query, feat_gallery, file_list, labels, clusters_means, rank=1):
    pass
