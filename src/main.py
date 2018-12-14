import numpy as np
import json
import yaml
from scipy.io import loadmat
from sklearn.cluster import KMeans
from train import set_feat_train, set_feat_train_valid
from test import rank_query, set_feat_test
from model import pca, optimize_metric


print('Reading YAML file...')
with open('../cfgs/conf.yml') as ymlfile:
    cfg = yaml.load(ymlfile)
for section in cfg:
    for attr in section.items():
        if attr[0] == 'BASE':
            rank = attr[1].get('RANK')
            bool_display = attr[1].get('DISPLAY')
            n_clusters = attr[1].get('N_CLUSTERS')
            n_clusters_valid = attr[1].get('N_CLUSTERS_VALID')
            n_clusters_test = attr[1].get('N_CLUSTERS_TEST')
            bool_transform = attr[1].get('TRANSFORM')
        elif attr[0] == 'METRIC':
            bool_metric_train = attr[1].get('METRIC_TRAIN')
            m_pca = attr[1].get('M_PCA')
        elif attr[0] == 'CLUSTERING':
            bool_cluster = attr[1].get('CLUSTER')
            n_init = attr[1].get('N_INIT')


print('Loading protocole data...')
cam_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
file_list = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()

gallery_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
gallery_idx = gallery_idx - 1

labels = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
labels = labels - 1

query_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
query_idx = query_idx - 1

train_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
train_idx = train_idx - 1

# Loading Features and Indices for Training, Query & Gallery
print('Loading feature data...')
with open('../pr_data/feature_data.json', 'r') as infile:
    features = json.load(infile)

if bool_transform:
    print('Flattening data...')
    features_proj = np.log(np.array(features) + 1)
    features = features_proj.tolist()

    if bool_metric_train:
        feat_train, train_idx, feat_valid, valid_idx = set_feat_train_valid(features, train_idx,
                                                                            n_clusters_valid, labels)

        print('Centering data...')
        mu_train, mu_valid = np.mean(np.array(feat_train), axis=0), np.mean(np.array(feat_valid), axis=0)

        print('Training metric...')
        print('-- Validating')
        NULL, n_iter = optimize_metric(np.array(feat_valid) - mu_valid[None, :], labels[valid_idx])

        print('-- Final Training')
        g_mat, n_iter = optimize_metric(np.array(feat_train) - mu_train[None, :],
                                        labels[train_idx], max_iter=n_iter, obj_f=1, alpha=1e-12)

        np.save('./metric_log_file.npy', g_mat)
        np.save('./train_mean.npy', mu_train)
    else:
        g_mat = np.load('./metric_log_file.npy', 'r')
        mu_train = np.load('./train_mean.npy', 'r')

    print('Applying metric on data...')
    features_proj = (np.array(features) - mu_train[None, :]).dot(g_mat.T)
    features = features_proj.tolist()

    feat_train = set_feat_train(features, train_idx)

    print('Applying PCA on data...')
    u_pca, mu_pca = pca(np.array(feat_train), m_pca=m_pca)
    features_proj = (np.array(features) - mu_pca[None, :]).dot(u_pca)
    features = features_proj.tolist()


print('Testing...')
rank_score = rank_query(features, query_idx, gallery_idx, file_list, labels, cam_idx, rank=rank, display=bool_display)
print('Rank score is %.2f' % rank_score)
print('Done with metric!')


if bool_cluster:
    print('Clustering testing set...')
    feat_test, test_idx = set_feat_test(features, query_idx, gallery_idx)

    k_means = KMeans(n_clusters=n_clusters_test, init='random', n_init=n_init, n_jobs=4)
    k_means.fit(feat_test)
    cluster_means = k_means.cluster_centers_

    if bool_transform:
        file_cluster_means_out = './cluster_means_transformed_file.npy'
    else:
        file_cluster_means_out = './cluster_means_file.npy'

    np.save(file_cluster_means_out, cluster_means)

    print('Testing...')
    rank_score = rank_query(features, query_idx, gallery_idx, file_list, labels, cam_idx, rank=rank,
                            display=bool_display, cluster_means=cluster_means)
    print('Rank score is %.2f' % rank_score)
