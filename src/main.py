import numpy as np
import json
import yaml
from scipy.io import loadmat
from sklearn.cluster import KMeans
from train import set_feat_train, set_feat_train_valid
from test import rank_query
from functions import cluster_means_sort


# Loading Features and Indices for Training, Query & Gallery
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

# Reading configurations file
print('Reading YAML file...')
with open('../cfgs/conf.yml') as ymlfile:
    cfg = yaml.load(ymlfile)
for section in cfg:
    for attr in section.items():
        if attr[0] == 'BASE':
            n_clusters = attr[1].get('N_CLUSTERS')
            n_clusters_valid = attr[1].get('N_CLUSTERS_VALID')
            train = attr[1].get('TRAIN')
            valid = attr[1].get('VALID')
            n_init = attr[1].get('N_INIT')
            disp = attr[1].get('DISPLAY')
            metric = attr[1].get('METRIC')

# Based on input from configuration file, decide whether to train or not.
if train:
    # If training, then based on input from configuration file, choose to apply validation or not.
    print('Training model...')
    if valid:
        # If applying validation, partition training data into training and validation sets.
        print('-- Applying validation...')
        feat_train, feat_valid, valid_idx = set_feat_train_valid(features, train_idx,
                                                                 n_clusters, n_clusters_valid, labels)

        # Apply K-Means on validation set.
        k_mean = KMeans(n_clusters=n_clusters_valid, init='random', n_init=2, n_jobs=2)
        k_mean.fit(feat_valid)

        # Get number of iterations for convergence of cluster means with validation set.
        n_iter = k_mean.n_iter_

        # Apply K-Means on entire training set with maximum iterations n_iter.
        print('-- Final training...')
        k_mean = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, n_jobs=4, max_iter=n_iter, tol=1e-6)
        k_mean.fit(feat_train)

    else:
        # If not applying validation, apply K-Means on entire training set.
        feat_train = set_feat_train(features, train_idx)

        k_mean = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, n_jobs=4, tol=1e-2)
        k_mean.fit(feat_train)

    # Save cluster means to .npy file in ./src folder.
    print('-- Saving cluster means...')
    print(k_mean.labels_)
    cluster_means = cluster_means_sort(k_mean.cluster_centers_, k_mean.labels_)
    np.save('./cluster_file.npy', cluster_means)

else:
    # If not training, load cluster means from .npy file in ./src folder.
    print('Loading cluster means...')
    cluster_means = np.array(np.load('./cluster_file.npy', 'r'))

# Test model with metric given in configuration file.
print('Testing...')
rank_query(features, query_idx, gallery_idx, file_list, labels, cluster_means, cam_id, rank=10)

print('Done!')
