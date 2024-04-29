import argparse
import collections
import os
import random
import scipy.io as si
import numpy as np
import pandas as pd
from SPC.density import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from code.algorithm.util import load_dataset


def sample(data: np.ndarray, label, ratio: float, c: int):
    or_data = data.copy()
    or_label = label.copy().reshape(-1,)
    or_label = np.array(or_label).reshape(-1, )
    random.seed(602)
    np.random.seed(602)
    c_label = c
    sub_label_idx = np.argwhere(or_label == c_label).reshape(-1, )
    sub_data = or_data[sub_label_idx]
    sub_label = or_label[sub_label_idx]
    or_data = np.delete(or_data, sub_label_idx, axis=0)
    or_label = np.delete(or_label, sub_label_idx, axis=0)
    nn = sub_data.shape[0]
    random_idx = np.random.choice(nn, np.floor(nn * ratio).astype(int), replace=False).reshape(-1, )
    sub_data_remain = np.delete(sub_data, random_idx, axis=0)
    sub_label_remain = np.delete(sub_label, random_idx, axis=0)
    or_data = np.concatenate([or_data, sub_data_remain], axis=0)
    or_label = np.concatenate([or_label, sub_label_remain], axis=0)
    return or_data, or_label

def compute_density(data: np.ndarray, label, args):
    data_tmp = data.copy()
    label -= np.min(label)
    label = np.array(label).reshape(-1,).astype(int)
    distance_series = pdist(data_tmp)
    distance_matrix = squareform(distance_series)
    distance_sort = np.sort(distance_matrix, axis=1)
    nn, dd = data_tmp.shape
    # radius_mean = np.zeros((data.shape[0],))
    # for i in range(data.shape[0]):
    #     radius_mean[i] = np.mean(distance_sort[i, 1:11])
    density = np.zeros((nn,))
    # for i in range(nn):
    #     for j in range(1, 20):
    #         density[i] += np.exp(-distance_sort[i, j])

    dc = np.mean(distance_sort[:, int(np.floor(0.02 * nn))])  # distance_sort[row, column]
    for i in range(nn):
        j = 1
        while j < nn and distance_sort[i, j] <= dc:
            density[i] += 1#np.exp(-distance_sort[i, j])
            j += 1

    label_set = np.unique(label).astype(int)
    avg_density = np.zeros((len(label_set),))
    num_label = np.zeros((len(label_set),))
    for ll in label_set:
        ll = int(ll)
        for i in range(nn):
            if label[i] == ll:
                avg_density[ll] += density[i]
                num_label[ll] += 1
    for i in range(len(avg_density)):
        avg_density[i] = avg_density[i] / num_label[i]
    return avg_density, label_set, density

def reset_label(y):
    y_set = set(y.tolist())
    hash_map = {}
    y_new = -np.ones((y.shape[0], ), dtype=int)
    for i, item in enumerate(y_set):
        idx = np.argwhere(y == item).reshape(-1, )
        y_new[idx] = i
    return y_new

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result/', type=str, help='the director to save results')
    parser.add_argument('--dir', default='../../data/', type=str, help='the director of dataset')
    parser.add_argument('--data_name', default='UNBALANCE4.csv', type=str,
                        help='dataset name, one of {overlap1, overlap2, birch1, '
                             'birch2, iris, breast, iris, wine, htru, knowledge}')
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--ratio', default=0, type=float)
    parser.add_argument('--show_fig', default=False, type=bool)
    parser.add_argument('--NoEnhance', default=False, type=bool)
    parser.add_argument('--has_label', default=True, type=bool)
    parser.add_argument('--resample', default=False, type=bool)
    args = parser.parse_args()
    multiple = 2
    or_name = args.data_name
    name, ty = args.data_name.split(".")

    args.data_name = or_name
    data, label = load_dataset(args, args.has_label, args.resample)
    data = data.astype(np.float32)
    print(data.shape)
    label_count = collections.Counter(label.reshape(-1, ).tolist())
    print(label_count)

    dddata = data.copy()
    for j in range(dddata.shape[1]):
        max_ = max(dddata[:, j])
        min_ = min(dddata[:, j])
        if max_ == min_:
            continue
        for i in range(dddata.shape[0]):
            dddata[i][j] = (dddata[i][j] - min_) / (max_ - min_)
    print(prompt + " after dilution")
    avg_density, label_set, density = compute_density(dddata, label, args)
    label_count = collections.Counter(label.reshape(-1, ).tolist())
    print(prompt, label_count)
    print(prompt, " average density: ", avg_density, label_set)

