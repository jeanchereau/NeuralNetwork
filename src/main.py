import numpy as np
import json
import yaml
from scipy.io import loadmat
from sklearn.cluster import KMeans
from src.train import set_feat_train, set_feat_train_valid
from src.test import rank_query


# Read configurations file in './cfgs'
print('Reading YAML file...')
with open('../cfgs/conf.yml') as ymlfile:
    cfg = yaml.load(ymlfile)
for section in cfg:
    for attr in section.items():
        if attr[0] == 'BASE':
            bool_display = attr[1].get('DISPLAY')
            rank = attr[1].get('RANK')
            metric = attr[1].get('METRIC')
            bool_cluster = attr[1].get('CLUSTER')
        if attr[0] == 'CLUSTERING':
            bool_train = attr[1].get('TRAIN')
            bool_valid = attr[1].get('VALID')
            n_clusters = attr[1].get('N_CLUSTERS')
            n_clusters_valid = attr[1].get('N_CLUSTERS_VALID')
            bool_pca_lda = attr[1].get('PCA_LDA')
            bool_kmeans = attr[1].get('KMEANS')
            n_init = attr[1].get('N_INIT')


# Loading Features and Indices for Training, Query & Gallery
print('Loading feature data...')
with open('../pr_data/feature_data.json', 'r') as jsonfile:
    features = json.load(jsonfile)

print('Loading indices...')
cam_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
file_list = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()

gallery_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
gallery_idx = gallery_idx - np.ones(gallery_idx.size, dtype=int)

labels = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
labels = labels - np.ones(labels.size, dtype=int)

query_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
query_idx = query_idx - np.ones(query_idx.size, dtype=int)

train_idx = loadmat('../pr_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
train_idx = train_idx - np.ones(train_idx.size, dtype=int)


if bool_cluster:
    # Based on input from configuration file, decide whether to train or not.
    if bool_train:

        if bool_pca_lda:
            features_set = features     # Perform PCA-LDA

            with open('../pr_data/feature_pca_lda_data.json', 'w') as outfile:
                json.dump(features_set, outfile)

            file_cluster_out = './cluster_file_pca_lda.npy'
        else:
            features_set = features
            file_cluster_out = './cluster_file.npy'

        # If training, then based on input from configuration file, choose to apply validation or not.
        print('Training model...')
        if bool_valid:
            # If applying validation, partition training data into training and validation sets.
            print('-- Applying validation...')
            feat_train, feat_valid, valid_idx = set_feat_train_valid(features_set, train_idx,
                                                                     n_clusters, n_clusters_valid, labels)

            # Apply K-Means on validation set.
            k_mean = KMeans(n_clusters=n_clusters_valid, init='random', n_init=2, n_jobs=2)
            k_mean.fit(feat_valid)

            # Get number of iterations for convergence of cluster means with validation set.
            n_iter = k_mean.n_iter_

            # Apply K-Means on entire training set with maximum iterations n_iter.
            print('-- Final training...')
            k_mean = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, n_jobs=4, max_iter=n_iter)
            k_mean.fit(feat_train)

        else:
            # If not applying validation, apply K-Means on entire training set.
            feat_train = set_feat_train(features, train_idx)

            k_mean = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, n_jobs=4)
            k_mean.fit(feat_train)

        # Save cluster means to .npy file in ./src folder.
        print('-- Saving cluster means...')
        cluster_means = k_mean.cluster_centers_
        np.save(file_cluster_out, cluster_means)

    else:
        # If not training, load cluster means from .npy file in ./src folder.
        print('Loading cluster means...')
        if bool_pca_lda:
            with open('../pr_data/feature_pca_lda_data.json', 'r') as jsonfile:
                features_set = json.load(jsonfile)

            file_cluster_in = './cluster_file_pca_lda.npy'
        else:
            features_set = features
            file_cluster_in = './cluster_file.npy'

        cluster_means = np.array(np.load(file_cluster_in, 'r'))
else:
    cluster_means = None
    features_set = features


# Test model with metric given in configuration file.
print('Testing...')
rank_score, ma_prec = rank_query(features_set, query_idx, gallery_idx, file_list, labels, cam_idx,
                                 cluster_means=cluster_means, rank=rank, metric=metric, display=bool_display)
print('Rank score is %.2f' % rank_score)
print('Mean Average Precision is %.df' % ma_prec)
print('Done!')
