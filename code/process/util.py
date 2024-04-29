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
            data = np.loadtxt(args.dir + args.data_name + "/" + args.data_name + ".txt")
        else:
            data = np.loadtxt(args.dir + args.data_name + "/" + args.data_name + "-" + str(args.ratio) + ".txt")
    elif ty == 'csv':
        if args.ratio == 0:
            data = pd.read_csv(args.dir + args.data_name + "/" + args.data_name + ".csv").values
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
    return data.astype(float), label

def process_data_glass(file_path, sep=','):
    data = pd.read_csv(file_path, sep=sep, header=None)
    print(data.head())
    data = data.iloc[:, 1:].values
    label = data[:, -1]
    data = data[:, :-1]
    return data, label

def process_data_abalone(file_path, sep=','):
    data = pd.read_csv(file_path, sep=sep, header=None).values
    print(data.shape)
    data = data[:, 1:]
    label = data[:, 0]
    data = data[:, 1:]
    label_set = set(label.tolist())
    new_label = np.zeros((label.shape[0], ), dtype=int)
    for i, item in enumerate(label_set):
        idx = np.argwhere(label == item).reshape(-1,)
        new_label[idx] = i
    return data, new_label

def process_data_balance(file_path, sep=','):
    data = pd.read_csv(file_path, sep=sep, header=None).values
    print(data.shape)
    label = data[:, 0]
    data = data[:, 1:]
    label_set = set(label.tolist())
    new_label = np.zeros((label.shape[0], ), dtype=int)
    for i, item in enumerate(label_set):
        idx = np.argwhere(label == item).reshape(-1,)
        new_label[idx] = i
    return data, new_label

def process_data_csv(file_path, sep=';', header=None, label_col=-1):
    if header is None:
        data = pd.read_csv(file_path, sep=sep, header=header)
    else:
        data = pd.read_csv(file_path, sep=sep)
    data = data.values[:, 1:]
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

def process_binary_and_label_category_feature(path, save_path, feature_idx_list):
    data = pd.read_csv(path)
    column = data.columns
    data = data.values
    for item in feature_idx_list:
        tmp = data[:, item].reshape(-1, )
        val_set = set(tmp.tolist())
        for i, val in enumerate(val_set):
            idxs = np.argwhere(tmp == val).reshape(-1, )
            data[idxs, item] = i
    data = pd.DataFrame(data, columns=column)
    data.to_csv(save_path, index=False)
    return data

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

def deleteClass(ll: list, or_data: np.ndarray, column) -> pd.DataFrame:
    data = or_data.copy()
    for item in ll:
        idx = np.argwhere(data[:, -1] == item).reshape(-1, )
        data = np.delete(data, idx, axis=0)
    label_count = collections.Counter(data[:, -1].reshape(-1, ).tolist())
    print(label_count)
    data = pd.DataFrame(data, columns=column)
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

# data = read_arff("../../data/phishing/PhishingData.arff", "../../data/phishing/phishing.csv")
# data = process_binary_and_label_category_feature("../../data/MaternalHealth/maternal_or.csv", "../../data/MaternalHealth/maternal.csv", [-1])
# data, label = process_data_csv("../../data/ecoli/ecoli.data", ',', None, -1)
# data = data[:, 1:]
# data = data.astype(float)
# label = label.astype(int)
# label_count = collections.Counter(label.reshape(-1, ).tolist())
# print(label_count)
# print(data[:5])
# save_data("../../data/ecoli/ecoli.csv", data, label)
# data, label = load_dataset_from_UCI(880)
# save_data("../../data/support2/support2.csv", data, label)

# data, label = load_dataset_from_UCI(159)
# label = label.reshape(-1, )
# print(collections.Counter(label.tolist()))
# save_data('../../data/gamma/gamma.csv', data, label)
# HCV 571
# ILPD 225
# diabetic 329
# support2 880
# shuttle 148
# Gamma telescope 159
# news popularity 332