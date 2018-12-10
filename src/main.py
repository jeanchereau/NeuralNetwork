import numpy as np
import json
import yaml
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from train import set_feat_train, set_feat_train_valid
from test import rank_query, set_feat_test
from model import pca, optimize_metric
from functions import f_measure


# Read configurations file in './cfgs'
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
            n_clusters_train = attr[1].get('N_CLUSTERS_TRAIN')
            n_clusters_test = attr[1].get('N_CLUSTERS_TEST')
            bool_transform = attr[1].get('TRANSFORM')
        elif attr[0] == 'METRIC':
            bool_metric_train = attr[1].get('METRIC_TRAIN')
            bool_kpca_train = attr[1].get('KPCA_TRAIN')
            bool_kpca = attr[1].get('KPCA')
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

if bool_transform:
    if bool_metric_train:
        # Loading Features and Indices for Training, Query & Gallery
        print('Loading feature data...')

        if bool_kpca:
            if bool_kpca_train:
                with open('../pr_data/feature_data.json', 'r') as infile:
                    features = json.load(infile)

                feat_train = np.array(set_feat_train(features, train_idx))
                mu = np.mean(feat_train, axis=0)

                print('Applying Kernel PCA...')
                k_pca = KernelPCA(n_components=m_pca, kernel='poly', n_jobs=4)
                k_pca.fit(feat_train - mu[None, :])

                alphas = k_pca.alphas_
                lambdas = k_pca.lambdas_
                d = np.sqrt(np.linalg.inv(np.diag(lambdas)))

                features_proj = np.array(features - mu[None, :]).dot((feat_train - mu[None, :]).T.dot(alphas.dot(d)))

                features = features_proj.tolist()

                with open('../pr_data/feature_kpca_data.json', 'w') as outfile:
                    json.dump(features, outfile)

            else:
                with open('../pr_data/feature_kpca_data.json', 'r') as infile:
                    features = json.load(infile)

            file_metric_out = './metric_kpca_file.npy'

        else:
            file_metric_out = './metric_file.npy'

        feat_train, train_idx, feat_valid, valid_idx = set_feat_train_valid(features, train_idx, n_clusters_valid, labels)

        print('Training metric...')
        print('-- Validating')
        g_mat, n_iter = optimize_metric(np.array(feat_valid), labels[valid_idx])

        print('-- Final Training')
        # g_mat, n_iter = optimize_metric(np.array(feat_train), labels[train_idx], max_iter=n_iter,
        # n_part=n_clusters_train // n_clusters_valid)

        np.save(file_metric_out, g_mat)

    else:

        print('Loading feature data...')
        if bool_kpca:
            file_features_in = '../pr_data/feature_kpca_data.json'
            file_metric_in = './metric_kpca_file.npy'
        else:
            file_features_in = '../pr_data/feature_data.json'
            file_metric_in = './metric_file.npy'

        with open(file_features_in, 'r') as infile:
            features = json.load(infile)

        # g_mat = np.load(file_metric_in, 'r')

    print('Applying metric on all features...')
    # features_proj = np.array(features).dot(g_mat.T)
    # features = features_proj.tolist()

else:
    with open('../pr_data/feature_data.json', 'r') as infile:
        features = json.load(infile)

print('Testing...')
rank_score, avg_prec, avg_recall = rank_query(features, query_idx, gallery_idx, file_list, labels, cam_idx,
                                              rank=rank, display=bool_display)
print('Rank score is %.2f' % rank_score)
print('Mean Average Precision is %.2f' % avg_prec)
print('Mean Average Recall is %.2f' % avg_recall)

f_measure(avg_prec, avg_recall)

print('Done with metric!')

if bool_cluster:
    print('Clustering testing set...')
    feat_test, test_idx = set_feat_test(features, query_idx, gallery_idx)

    k_means = KMeans(n_clusters=n_clusters_test, init='random', n_init=n_init, n_jobs=4)
    k_means.fit(feat_test)
    cluster_means = k_means.cluster_centers_

    if bool_kpca:
        file_cluster_means_out = './cluster_means_kpca_file.npy'
    else:
        file_cluster_means_out = './cluster_means_file.npy'

    np.save(file_cluster_means_out, cluster_means)

else:
    if bool_kpca:
        file_cluster_means_in = './cluster_means_kpca_file.npy'
    else:
        file_cluster_means_in = './cluster_means_file.npy'

    cluster_means = np.array(np.load(file_cluster_means_in, 'r')).tolist()

print('Testing...')
rank_score, avg_prec, avg_recall = rank_query(features, query_idx, gallery_idx, file_list, labels, cam_idx,
                                              rank=rank, display=bool_display, cluster_means=cluster_means)
print('Rank score is %.2f' % rank_score)
print('Mean Average Precision is %.2f' % avg_prec)
print('Mean Average Recall is %.2f' % avg_recall)

f_measure(avg_prec, avg_recall)

print('Done with metric!')
