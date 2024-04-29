import os

import scipy.io as si
import collections

import pandas as pd
from scipy.io import arff
import numpy as np

def load_dataset(args, with_label=True, unbalance=False, ty='csv'):
    if args.data_name.__contains__('.'):
        name, ty = args.data_name.split(".")
        args.data_name = name
    if ty == 'txt':
        if args.ratio == 0:
            data = np.loadtxt(args.dir + args.data_name + "/" + args.data_name + ('-resample.txt' if unbalance else '.txt'))
        else:
            data = np.loadtxt(args.dir + args.data_name + "/" + args.data_name + "-" + str(args.ratio) + ".txt")
    elif ty == 'csv':
        if args.ratio == 0:
            data = pd.read_csv(args.dir + args.data_name + "/" + args.data_name + ('-resample.csv' if unbalance else ".csv")).values
        else:
            data = pd.read_csv(args.dir + args.data_name + "/" + args.data_name + "-" + str(args.ratio) + ".csv").values
    elif ty == 'mat':
        data = si.loadmat(args.dir + args.data_name + "/" + args.data_name + ".mat")
        X = data['X']
        y = data['y']
        data = np.concatenate([X, y], axis=1)
    if with_label:
        label = data[:, -1].astype(int)
        data = data[:, :-1]
    else:
        label = np.loadtxt(args.dir + args.data_name + "/" + args.data_name + '-label.txt').astype(int).reshape(-1, )
    return data, label

def process_data_csv(file_path, sep=';', header=None, label_col=-1):
    if header is None:
        data = pd.read_csv(file_path, sep=sep, header=header)
    else:
        data = pd.read_csv(file_path, sep=sep)
    data = data.values
    label = data[:, label_col]
    if label_col == -1:
        data = data[:, :label_col]
    else:
        data = data[:, label_col + 1:]
    label_set = set(label.tolist())
    print(label_set)
    new_label = np.zeros((label.shape[0], ), dtype=int)
    for i, item in enumerate(label_set):
        idx = np.argwhere(label == item).reshape(-1,)
        new_label[idx] = i
    return data, new_label

def save_data(file_path, X, y):
    column = [('dim' + str(i)) for i in range(X.shape[1])]
    column.append('y')
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    data = pd.DataFrame(data, columns=column)
    data.to_csv(file_path, index=False)

def read_arff(path, save):
    df = arff.loadarff(path)
    data = pd.DataFrame(df[0])
    data = data.astype(float)
    label_name = data['Result'].values
    label_set = set(label_name)
    label = np.zeros((data.shape[0], ), dtype=int)
    for i, item in enumerate(label_set):
        idxs = np.argwhere(label_name == item).reshape(-1, )
        label[idxs] = i
    # for i in range(data.shape[1] - 1):
    #     column.append('dim' + str(i))
    # column.append('y')
    # data.columns = column
    data['Result'] = label
    data['Result'] = data['Result'].astype(int)
    data.to_csv(save, index=False)
    return data

def computePurity(labels_true, labels_pred):
  clusters = np.unique(labels_pred)
  labels_true = np.reshape(labels_true, (-1, ))
  labels_pred = np.reshape(labels_pred, (-1, ))
  count = []
  for c in clusters:
    idx = np.where(labels_pred == c)
    labels_tmp = labels_true[idx].reshape((-1, )).astype(int)
    count.append(np.bincount(labels_tmp).max())
  return np.sum(count) / labels_true.shape[0]

def calculate_metrix(labels, pred, args, clustering_time, algorithm_name, clustering_name, para_map):
    # clustering by kmeans
    from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, \
    fowlkes_mallows_score as FMI, homogeneity_completeness_v_measure as HCV

    ari = ARI(labels, pred)
    nmi = AMI(labels, pred)
    fmi = FMI(labels, pred)
    purity = (computePurity(labels, pred))
    hcv = HCV(labels, pred)
    homogeneity = (hcv[0])
    completeness = (hcv[1])
    vm = (hcv[2])
    my_metrics = [[ari, nmi, fmi, purity, vm, homogeneity, completeness, clustering_time]]
    my_column = ['ari', 'nmi', 'fmi', 'purity', 'vm', 'homogeneity', 'completeness', 'time']
    key_list = []
    val_list = []
    for key, val in para_map.items():
        key_list.append(key)
        val_list.append(val)
    if len(key_list) > 0:
        my_column.extend(key_list)
        my_metrics[0].extend(val_list)
    # print(my_metrics)
    # print(my_column)
    my_metrics = pd.DataFrame(my_metrics, columns=my_column)
    save_path = args.save_dir + args.data_name
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    save_path = save_path + "/result-" + str(args.ratio) + "-" + algorithm_name + "-" + clustering_name + ".csv"
    if os.path.exists(save_path):
        my_metrics.to_csv(save_path, index=False, header=None, mode='a')
    else:
        my_metrics.to_csv(save_path, index=False)

def calculate_metric(labels, pred, args, clustering_time, algorithm_name, clustering_name, para_map):
    # clustering by kmeans
    from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, \
    fowlkes_mallows_score as FMI, homogeneity_completeness_v_measure as HCV
    ari = ARI(labels, pred)
    nmi = AMI(labels, pred)
    fmi = FMI(labels, pred)
    purity = (computePurity(labels, pred))
    hcv = HCV(labels, pred)
    homogeneity = (hcv[0])
    completeness = (hcv[1])
    vm = (hcv[2])
    my_metrics = [[ari, nmi, fmi, purity, vm, homogeneity, completeness, clustering_time]]
    my_column = ['ari', 'nmi', 'fmi', 'purity', 'vm', 'homogeneity', 'completeness', 'time']
    key_list = []
    val_list = []
    for key, val in para_map.items():
        key_list.append(key)
        val_list.append(val)
    if len(key_list) > 0:
        my_column.extend(key_list)
        my_metrics[0].extend(val_list)
    return my_metrics

def load_dataset_from_UCI(id: int):
    from ucimlrepo import fetch_ucirepo
    # fetch dataset
    zoo = fetch_ucirepo(id=id)
    # data (as pandas dataframes)
    X = zoo.data.features
    y = zoo.data.targets
    # metadata
    print(X.shape, type(X))
    # variable information
    print(y.shape, type(y))
    return X.values, y.values

def reset_label(y):
    y_set = set(y.tolist())
    hash_map = {}
    y_new = -np.ones((y.shape[0], ), dtype=int)
    for i, item in enumerate(y_set):
        idx = np.argwhere(y == item).reshape(-1, )
        y_new[idx] = i
    return y_new

# HCV 571
# ILPD 225
# diabetic 329
# support2 880