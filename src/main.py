import json
import yaml
from scipy.io import loadmat
from sklearn.cluster import KMeans
from train import set_feat_train, set_feat_train_valid
from test import set_feat_query_gallery


with open('../pr_data/feature.json', 'r') as jsonfile:
    features = json.load(jsonfile)

cam_id = loadmat('../pr_data/cukh03_new_protocol_config_labeled.mat')['camId'].flatten()
file_list = loadmat('../pr_data/cukh03_new_protocol_config_labeled.mat')['filelist'].flatten()
gallery_idx = loadmat('../pr_data/cukh03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
labels = loadmat('../pr_data/cukh03_new_protocol_config_labeled.mat')['labels'].flatten()
query_idx = loadmat('../pr_data/cukh03_new_protocol_config_labeled.mat')['query_idx'].flatten()
train_idx = loadmat('../pr_data/cukh03_new_protocol_config_labeled.mat')['train_idx'].flatten()

with open('../cfgs/conf.yml') as ymlfile:
    cfg = load(ymlfile)
for section in cfg:
    for attr in section.items():
        if attr[0] == 'BASE':
            n_clusters = attr[1].get('n_clusters')
            n_clusters_valid = attr[1].get('n_clusters_valid')
            valid = attr[1].get('valid')
            n_init = attr[1].get('n_init')

if valid:
    feat_train, feat_valid = set_feat_train_valid(features, train_idx, n_clusters, n_clusters_valid, labels)

    k_means = KMeans(n_clusters=n_clusters_valid, init='random', n_init=2, n_jobs=2)
    k_means.fit(feat_valid)

    n_iter = k_means.n_iter_

    k_means = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, n_jobs=2, max_iter=n_iter)
    k_means.fit(feat_train + feat_valid)

else:
    feat_train = set_feat_train(features, train_idx)

    k_means = KMeans(n_clusters=n_clusters, init='random', n_init=n_init, n_jobs=2)
    k_means.fit(feat_train)

cluster_means = k_means.cluster_centers_

feat_query, feat_gallery = set_feat_query_gallery(features, query_idx, gallery_idx)

# rank_query(rank=5)
