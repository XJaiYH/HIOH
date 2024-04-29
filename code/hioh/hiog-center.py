import argparse
import os
import time

from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from util import load_dataset, calculate_metrix

os.environ["OMP_NUM_THREADS"] = '1'
global visit
global complete
complete = False

def plotCluster(data: np.ndarray, labels: np.ndarray, title: str, args):
    fig, ax = plt.subplots()
    label_set = set(labels.tolist())
    color = [
        "#900C3F",  # 紫红色
        "#006400",  # 深绿色
        "#4B0082",  # 靛青色
        "#FF4500",  # 橙红色
        "#FF1493",  # 深粉色
        "#008B8B",  # 深青色
        "#FF7F50",  # 珊瑚色
        "#4682B4",  # 钢蓝色
        "#A9A9A9",  # 暗灰色
        "#556B2F",  # 暗绿色
        "#9370DB",  # 中紫色
        "#8B7355",  # 赭色
        "#FFD700",  # 库金色
        "#2E8B57",  # 海洋绿色
        "#008B8B",  # 暗藏青色
        "#BDB76B",  # 黄褐色
        "#654321",  # 深棕色
        "#9400D3",  # 暗紫色
        "#008080",  # 暗青色
        "#CD5C5C"  # 褐红色
    ]
    lineform = ['o']
    for i in label_set:
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        fontSize = 30
        colorNum = i % len(color)
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(title, fontsize=20)
    save_path = args.save_dir + args.data_name
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    save_path += "/" + title + ".png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plotCircle(data: np.ndarray, labels: np.ndarray, radius):
    fig, ax = plt.subplots()
    label_set = set(labels)
    color = [
        "#900C3F",  # 紫红色
        "#006400",  # 深绿色
        "#4B0082",  # 靛青色
        "#FF4500",  # 橙红色
        "#FF1493",  # 深粉色
        "#008B8B",  # 深青色
        "#FF7F50",  # 珊瑚色
        "#4682B4",  # 钢蓝色
        "#A9A9A9",  # 暗灰色
        "#556B2F",  # 暗绿色
        "#9370DB",  # 中紫色
        "#8B7355",  # 赭色
        "#FFD700",  # 库金色
        "#2E8B57",  # 海洋绿色
        "#008B8B",  # 暗藏青色
        "#BDB76B",  # 黄褐色
        "#654321",  # 深棕色
        "#9400D3",  # 暗紫色
        "#008080",  # 暗青色
        "#CD5C5C"  # 褐红色
    ]
    lineform = ['o']
    for i in label_set:
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        fontSize = 30
        colorNum = i % len(color)
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for i in range(data.shape[0]):
        draw_circle = plt.Circle(data[i], radius[i], fill=False)
        ax.add_artist(draw_circle)
    plt.show()

def boost(data_ori:np.ndarray, data: np.ndarray, data_idx, previous_map, args, density, radius)->np.ndarray:
    '''
    :param data_ori: the original dataset
    :param data: the dataset beginning the t-th iteration
    :param data_idx: the index of objects in original dataset
    :param previous_map: key is the object index, and value is a list that contains indexes of objects that hop to the key object, so we can know where the each object in the original dataset hop to.
    :param args:
    :param density: the density of each object
    :param radius: the radius of each object
    :return:
    '''
    if args.verbose:
        print(prompt + "ameliorate the dataset")
    data_tmp = data.copy()
    tree = KDTree(data_tmp)
    distance_sort, sort_idx = tree.query(data_tmp, k=min(args.k+1, data_tmp.shape[0]))
    nn, dd = data_tmp.shape

    ##### # check whether achieving the ending conditions # #####
    is_over = True
    for i in range(nn):
        j = 1
        while j < nn and distance_sort[i, j] < radius[data_idx[i]]:
            if density[data_idx[sort_idx[i, j]]] > density[data_idx[i]]:
                is_over = False
                break
            j += 1
        if is_over is False:
            break
        else:
            continue
    if is_over:
        global complete
        complete = True
        return data_tmp, data_idx, previous_map, []

    if args.show_fig:
        print(prompt + "visualize the dataset of each iteration")
        if dd > 2:
            data_pca = pca.fit_transform(data_tmp)
            plotCircle(data_pca, label[data_idx], radius[data_idx])
        else:
            plotCircle(data_tmp, label[data_idx], radius[data_idx])

    ##### # OBJECTS HOPPING # #####
    hashmap = {}
    for i in range(nn):
        j = 1
        density_center_idx = -1
        ### # calculate high-density set # ###
        id_list = []
        while j < min(args.k+1, nn) and distance_sort[i, j] <= radius[data_idx[i]]:
            if density[data_idx[sort_idx[i, j]]] > density[data_idx[i]]:
                id_list.append(sort_idx[i, j])
            j += 1

        min_dis = np.inf
        ### # find high-density-center in the neighborhood of xi # ###
        for item in id_list:
            tmp_dis = np.sum(
                np.sqrt(np.sum(np.square(data_tmp[item] - data_tmp[id_list]), axis=1)))  # distance_matrix[item, jj]

            # for jj in range(len(id_list)):
            #     tmp_dis += np.sqrt(np.sum(np.square(data_tmp[item] - data_tmp[jj])))#distance_matrix[item, jj]
            if tmp_dis < min_dis:
                density_center_idx = item
                min_dis = tmp_dis
        ### # HOPPING # ###
        if density_center_idx != -1:
            if hashmap.get(data_idx[density_center_idx]) is None:
                hashmap[data_idx[density_center_idx]] = []
            if visit[data_idx[i]] is False:
                visit[data_idx[i]] = True
                hashmap[data_idx[density_center_idx]].append(int(data_idx[i]))
            hashmap[data_idx[density_center_idx]].extend(previous_map.get(data_idx[i], []))
        else:
            if hashmap.get(data_idx[i]) is None:
                hashmap[data_idx[i]] = previous_map.get(data_idx[i], [])
            else:
                hashmap[data_idx[i]].extend(previous_map.get(data_idx[i], []))

    # remove overlapping objects, and remain objects without overlapping
    new_data = np.zeros((len(hashmap), dd))
    data_idx = [-1 for i in range(len(hashmap))]
    idx_map = {}
    for i, key in enumerate(hashmap.keys()):
        new_data[i] = data_ori[int(key)]
        data_idx[i] = int(key)
        idx_map[key] = i
    return new_data, data_idx, hashmap, idx_map

from sklearn.metrics import adjusted_mutual_info_score

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result/', type=str, help='the director to save results')
    parser.add_argument('--dir', default='../../data/', type=str, help='the director of dataset')
    parser.add_argument('--data_name', default='tae.csv', type=str,
                        help='dataset name, one of {overlap1, overlap2, birch1, '
                             'birch2, iris, breast, iris, wine, htru, knowledge}')
    parser.add_argument('--ratio', default=1, type=int)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--has_label', default=True, type=bool)
    parser.add_argument('--show_fig', default=False, type=bool)
    parser.add_argument('--NoEnhance', default=False, type=bool)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--resample', default=False, type=bool)
    k_set = [i for i in range(15, 71)]
    args = parser.parse_args()

    algo_name = 'HIOH'
    prompt = "【prompt information】 "
    if os.path.isdir(args.save_dir) == False:
        os.mkdir(args.save_dir)
    or_name = args.data_name
    name, ty = args.data_name.split(".")

    args.data_name = or_name
    data, label = load_dataset(args, args.has_label, args.resample)
    cluster_num = len(set(label.tolist()))
    print(data.shape)
    print(prompt + "Loading dataset {}, shape:{}*{}, ratio:{}".format(name, data.shape[0], data.shape[1], args.ratio))

    pca = PCA(n_components=2)

    from sklearn.cluster import AgglomerativeClustering

    ########################normalization, this is not necessary###################################
    data_without_nml = data.copy()
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)

    res_kmeans = KMeans(n_clusters=len(set(label))).fit(data)
    print(prompt + "kmeans clustering results for ori dataset：", adjusted_mutual_info_score(label, res_kmeans.labels_))
    start = time.time()
    agg_res = AgglomerativeClustering(n_clusters=len(set(label))).fit(data)
    agg_nmi = adjusted_mutual_info_score(label, agg_res.labels_)
    print(prompt + "agg clustering results for ori dataset：", agg_nmi)
    print('time: ', time.time() - start)

    # if data.shape[1] > 2:
    #     data_pca = pca.fit_transform(data)
    #     plotCluster(data_pca, label, title="original dataset", args=args)
    # else:
    #     plotCluster(data, label, title="original dataset", args=args)

    nn, dd = data.shape
    for kk in k_set:
        args.k = kk
        print("****************************** # k is {} # ******************************".format(args.k))
        total_time = 0
        start = time.time()
        tree = KDTree(data)
        distance_sort, sort_idx = tree.query(data, args.k + 1)
        radius_mean = np.zeros((data.shape[0], ))
        for i in range(data.shape[0]):
            radius_mean[i] = np.mean(distance_sort[i, 1:args.k+1])
        density = np.zeros((nn,))
        for i in range(nn):
            density[i] = np.exp(-radius_mean[i])

        # calculate the mean density of each neighborhood
        mean_density = np.zeros((nn,))
        for i in range(nn):
            mean_density[i] += density[i]
            j = 1
            while j <= args.k and distance_sort[i, j] <= radius_mean[i]:
                mean_density[i] += density[sort_idx[i, j]]
                j += 1
            mean_density[i] /= j

        # calculate the mean density of each object (averaging density progress)
        new_density = np.zeros((nn,))
        for i in range(nn):
            j = 1
            new_density[i] = density[i]
            while j <= args.k and distance_sort[i, j] <= radius_mean[i]:
                new_density[i] += mean_density[sort_idx[i, j]]
                j += 1
            new_density[i] /= j
            new_density[i] += np.random.random() / 1e6
        density = new_density

        previous_map = {}
        data_idx = [i for i in range(data.shape[0])]
        data_boost = data.copy()
        data_ori = data.copy()
        data_boost_full = data.copy()
        visit = [False for i in range(data.shape[0])]
        complete = False
        total_time += time.time() - start

        print(prompt + " in boosting!")
        itr = 1
        while True:
            itr += 1
            start = time.time()
            data_boost, data_idx, previous_map, idx_map = boost(data_ori, data_boost, data_idx, previous_map, args, density, radius_mean)
            if complete:
                break
            # print(prompt + "data_boost shape: ", data_boost.shape, len(data_idx))
            # print(prompt + "previous shape: ", len(previous_map))
            # print(prompt + "after boost function")
            sum = 0
            null_num = 0
            for i, (key, val) in enumerate(previous_map.items()):
                sum += len(val)
                if len(val) > 0:
                    data_boost_full[val] = data_boost[int(idx_map[key])]
                else:
                    null_num += 1
            # print(prompt + "sum: ", sum, "null_sum: ", null_num)
            total_time += time.time() - start
            # data_tmp_save = np.concatenate([data_boost_full, label.reshape(-1, 1)], axis=1)
            # np.savetxt(args.save_dir + args.data_name + "/ameliorated-" + str(kk) + ".txt", data_tmp_save)

            if args.show_fig:
                if data_boost.shape[1] > 2:
                    data_pca = pca.fit_transform(data_boost)
                    plotCluster(data_pca, label[data_idx], title=('NoEnhance-3' if args.NoEnhance is True else '') + "dataset after boosting-" + str(itr),
                                args=args)
                else:
                    plotCluster(data_boost, label[data_idx],
                                title=('NoEnhance-3' if args.NoEnhance is True else '') + "dataset after boosting-" + str(itr), args=args)

                if data_boost.shape[1] > 2:
                    data_pca = pca.fit_transform(data_boost_full)
                    plotCluster(data_pca, label, title="dataset after boosting compare to ori-" + str(itr), args=args)
                else:
                    plotCluster(data_boost_full, label, title="dataset after boosting compare to ori-" + str(itr), args=args)
        print(prompt + " after boosting! total time: {}".format(total_time))

        data_copy = data_boost_full
        # data_tmp_save = np.concatenate([data_boost_full, label.reshape(-1, 1)], axis=1)
        # np.savetxt(args.save_dir + args.data_name + "/ameliorated-" + str(kk) + ".txt", data_tmp_save)

        print(data_boost.shape)
        # kmeans clustering
        if args.verbose:
            print(prompt + " executing kmeans clustering")
        kmeans_res = KMeans(n_clusters=cluster_num).fit_predict(data_copy)
        para_map = {}
        calculate_metrix(label, kmeans_res, args, total_time, algo_name, 'kmeans', para_map)

        #agg clustering
        if args.verbose:
            print(prompt + " executing agg clustering")
        agg_res = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(data_copy)
        para_map = {}
        calculate_metrix(label, agg_res, args, total_time, algo_name, 'agg', para_map)

        if args.show_fig:
            if data_boost.shape[1] > 2:
                data_pca = pca.fit_transform(data_boost_full)
                plotCluster(data_pca, label, title="dataset after boosting compare to ori-" + str(itr), args=args)
            else:
                plotCluster(data_boost_full, label, title="dataset after boosting compare to ori-" + str(itr), args=args)