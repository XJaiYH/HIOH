# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:04:58 2022

@author: 佡儁
"""
import argparse
import os.path
import time

import numpy as np
import scipy.spatial.distance as dis
from sklearn.cluster import Birch, SpectralClustering, DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import KDTree
from tqdm import tqdm
from code.hioh.util import load_dataset, calculate_metrix

def shrink(data, num, step_length):
    bata = data.copy()
    dim = data.shape[0]
    tree = KDTree(data)
    distance_sort, distance_index = tree.query(data, max(num+2, round(dim*0.015)+1))
    # distance = dis.pdist(data)
    # distance_matrix = dis.squareform(distance)
    # distance_sort = np.sort(distance_matrix, axis=1)
    # distance_index = np.argsort(distance_matrix, axis=1)
    area = np.mean(distance_sort[:, round(dim*0.015)])
    density = np.zeros(dim)
    count = tree.query_radius(data, area, count_only=True)
    for i in range(dim):
        # listss = 1
        # while (distance_sort[i,listss]<area):
        #     listss = listss + 1
        density[i] = count[i]
        # density[i] = listss
    density_mean = np.mean(density)
    density_yuzhi = density_mean

    buchang = np.mean(distance_sort[:, 1])
    for i in range(dim):
        if density[i] >= density_yuzhi:
            list = []
        else:
            list = distance_index[i, 1:(num+1)]

        linshi = np.zeros(data.shape[1],dtype=np.float32)
        ss = 1
        for j in list:
            if (data[j] == data[i]).all():
                ss = ss + 1
            else:
                ff = ((data[j] - data[i]))
                fff = (distance_sort[i, 1] / (distance_sort[i, ss]*distance_sort[i, ss]))
                linshi = linshi + ff * buchang * fff
                ss = ss + 1
        bata[i] = data[i]+linshi*step_length
    return bata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result-intro/', type=str, help='the director to save results')
    parser.add_argument('--dir', default='../../data/', type=str, help='the director of dataset')
    parser.add_argument('--data_name', default='htru.csv', type=str,
                        help='dataset name, one of {overlap1, overlap2, birch1, '
                             'birch2, iris, breast, iris, wine, htru, knowledge}')
    parser.add_argument('--ratio', default=1, type=int)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--show_fig', default=False, type=bool)
    parser.add_argument('--NoEnhance', default=False, type=bool)
    parser.add_argument('--resample', default=False, type=bool)
    args = parser.parse_args()
    print(args.resample)
    data, label = load_dataset(args, True, args.resample)
    # label = np.zeros((data.shape[0], ), dtype=int)
    prompt = "【prompt information】 "
    if os.path.isdir(args.save_dir + '/' + args.data_name) is False:
        os.mkdir(args.save_dir + '/' + args.data_name)

    ######### # normalization # ###############
    data_without_nml = data.copy()
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)

    cluster_num = len(set(label.tolist())) #    [1,1,2,3,2] [1,2,3]

    T_set = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]#bank会稍微大点
    # T_set = [0.5]
    K_set = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    # K_set = [24]
    iterationTime = 10

    with tqdm(total=len(T_set) * len(K_set) * 10) as tbar:
        for k in K_set:
            for t in T_set:
                data_copy = data.copy()
                total_time = 0

                for j in range(iterationTime):
                    start = time.time()
                    data_copy = shrink(data_copy, k, t)
                    total_time += time.time() - start

                    # kmeans
                    if args.verbose:
                        print(prompt + " executing kmeans clustering")
                    kmeans_res = KMeans(n_clusters=cluster_num, n_init='auto').fit_predict(data_copy)
                    para_map = {'K': k,
                                'T': t,
                                'D': j}
                    calculate_metrix(label, kmeans_res, args, total_time, 'HIBOG', 'kmeans', para_map)

                    # agg
                    if args.verbose:
                        print(prompt + " executing agg clustering")
                    agg_res = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(data_copy)
                    para_map = {'K': k,
                                'T': t,
                                'D': j}
                    calculate_metrix(label, agg_res, args, total_time, 'HIBOG', 'agg', para_map)

                    tbar.update(1)
                # np.savetxt(args.save_dir + args.data_name + '/ameliorated.txt', data_tmp)


