import numpy as np
import json
import yaml
from scipy.io import loadmat
from sklearn.cluster import KMeans
from train import set_feat_train, set_feat_train_valid
from test import rank_query


print('Loading data...')
with open('../pr_data/feature_data.json', 'r') as jsonfile:
    features = json.load(jsonfile)

cam_id = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
file_list = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()

gallery_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
gallery_idx = gallery_idx - np.ones(gallery_idx.size, dtype=int)

labels = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
labels = labels - np.ones(labels.size, dtype=int)

query_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
query_idx = query_idx - np.ones(query_idx.size, dtype=int)

train_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
train_idx = train_idx - np.ones(train_idx.size, dtype=int)

print('Reading YAML file...')
with open('../cfgs/conf.yml') as ymlfile:
    cfg = yaml.load(ymlfile)
for section in cfg:
    for attr in section.items():
        if attr[0] == 'BASE':
            n_clusters = attr[1].get('n_clusters')
            n_clusters_valid = attr[1].get('n_clusters_valid')
            train = attr[1].get('train')
            valid = attr[1].get('valid')
            n_init = attr[1].get('n_init')

if train:
    print('Training model...')
    if valid:
        feat_train, sub_train_idx, feat_valid, valid_idx = set_feat_train_valid(features, train_idx, n_clusters,
                                                                                n_clusters_valid, labels)

        k_means = KMeans(n_clusters=n_clusters_valid, init='random', n_init=2, n_jobs=2)
        k_means.fit(feat_valid)

        n_iter = k_means.n_iter_

        k_means = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, n_jobs=3, max_iter=n_iter)
        k_means.fit(feat_train + feat_valid)
    else:
        feat_train = set_feat_train(features, train_idx)

        k_means = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, n_jobs=3)
        k_means.fit(feat_train)

    cluster_means = k_means.cluster_centers_
    cluster_labels = k_means.labels_

    np.save('./cluster_means_file.npy', cluster_means)
    np.save('./cluster_labels_file.npy', cluster_labels)
else:
    cluster_means = np.load('./cluster_means_file.npy', 'r')
    cluster_labels = np.load('./cluster_labels_file.npy', 'r')

print(cluster_labels)

print('Testing...')
rank_query(features, query_idx, gallery_idx, file_list, labels, cluster_means, cluster_labels, cam_id, rank=10)

print('Done!')
