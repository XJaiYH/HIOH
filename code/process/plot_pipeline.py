'''
对簇进行下采样，即保留a%的原样本
'''
import collections

import matplotlib.pyplot as plt
import numpy as np
import random

import pandas as pd
from sklearn.neighbors import KDTree

from code.dm.DM import plotCluster

global kk
kk = 16

def plotCluster(data: np.ndarray, labels: np.ndarray, title: str, idx=None, dis_idx=None, dis_sort=None, radius=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    label_set = set(labels.tolist())
    color = [
        '#606470', # 灰色
        '#6499E9',
        #'#db6400', # 淡蓝色
        '#F52F2F', #茄红色
        "#900C3F",  # 紫红色
        '#9DBDF5', # 橙色
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
    fontSize = 135
    for i in label_set:
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        colorNum = int(i % len(color))
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum], zorder=2)
    if idx is not None and dis_idx is not None:
        global kk
        idxs = dis_idx[idx, 1:].reshape(-1, )
        print(idxs.shape)
        for i in range(kk):
            plt.plot([data[idx, 0], data[idxs[i], 0]], [data[idx, 1], data[idxs[i], 1]],
                     color="#008080", zorder=1, linewidth=3)
        j = 1
        while radius >= dis_sort[idx, j]:
            plt.scatter(data[idxs[j-1], 0], data[idxs[j-1], 1], fontSize, color[1], lineform[0], zorder=2)
            j += 1
    if idx is not None:
        plt.scatter(data[idx, 0], data[idx, 1], fontSize + 10, color[2], lineform[0], zorder=2)
    ax.set_ylim((-0.05, 0.85))
    ax.set_xlim((-0.05, 1.05))
    ax.set_xticks([])
    ax.set_yticks([])
    if radius is not None:
        draw_circle = plt.Circle((data[idx, 0], data[idx, 1]), radius, fill=False, color='#008080', linewidth=3)
        ax.add_artist(draw_circle)
    save_path = '../../data/aggSample/' + title + '.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plotEN(data: np.ndarray, labels: np.ndarray, title: str, idx=None, idx_set=None, nn_list=None, radius=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    label_set = set(labels.tolist())
    color = [
        '#606470', # 灰色
        '#6499E9', # 蓝色
        '#F52F2F', # 茄红色
        '#db6400', # 橙色
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
    fontSize = 135
    for i in label_set:
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        colorNum = int(i % len(color))
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    if idx is not None:
        for item in idx_set:
            # plt.plot([data[idx, 0], data[item, 0]], [data[idx, 1], data[item, 1]],
            #          color="#008080", zorder=1, linewidth=2.5)
            plt.scatter(data[item, 0], data[item, 1], fontSize, color[1], lineform[0], zorder=2)
        # for item in nn_list:
        #     plt.scatter(data[item, 0], data[item, 1], fontSize, color[2], lineform[0], zorder=2)
        plt.scatter(data[idx, 0], data[idx, 1], fontSize+10, color[2], lineform[0], zorder=2)
    ax.set_ylim((-0.05, 0.85))
    ax.set_xlim((-0.05, 1.05))
    ax.set_xticks([])
    ax.set_yticks([])
    # for item in nn_list:
    #     draw_circle = plt.Circle((data[item, 0], data[item, 1]), radius[item], fill=False, color='#008080', linewidth=2.5)
    #     ax.add_artist(draw_circle)
    save_path = '../../data/aggSample/extend_neighbor.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_EN(data: np.ndarray, labels: np.ndarray, title: str, idx=None, nn_list=None, radius=None, dis_sort=None, dis_idx=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    label_set = set(labels.tolist())
    color = [
        '#606470', # 灰色
        '#db6400', # 橙色
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
    fontSize = 135
    for i in label_set:
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])
        colorNum = int(i % len(color))
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    if idx is not None:
        for item in nn_list:
            # plt.plot([data[idx, 0], data[item, 0]], [data[idx, 1], data[item, 1]],
            #          color="#008080", zorder=1, linewidth=2.5)
            plt.scatter(data[item, 0], data[item, 1], fontSize, color[1], lineform[0], zorder=2)
            t = 1
            # while radius[item] >= dis_sort[item, t]:
            #     point = data[dis_idx[item, t]]
            #     plt.plot([data[item, 0], point[0]], [data[item, 1], point[1]], color='#008080', zorder=1, linewidth=2.5)
            #     t += 1
        plt.scatter(data[idx, 0], data[idx, 1], fontSize+10, color[2], lineform[0], zorder=2)
    ax.set_ylim((-0.05, 0.85))
    ax.set_xlim((-0.05, 1.05))
    ax.set_xticks([])
    ax.set_yticks([])
    for item in nn_list:
        draw_circle = plt.Circle((data[item, 0], data[item, 1]), radius[item], fill=False, color='#008080', linewidth=2.5)
        ax.add_artist(draw_circle)
    draw_circle = plt.Circle((data[idx, 0], data[idx, 1]), radius[idx], fill=False, color='#008080', linewidth=2.5)
    ax.add_artist(draw_circle)
    save_path = '../../data/aggSample/EN_diagraph.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_radius(file_path="../../data/aggSample/aggSample-6.txt"):
    global kk
    data = np.loadtxt(file_path)
    # data = pd.read_csv(file_path).values
    label = data[:, -1]
    data = data[:, :-1]
    # for j in range(data.shape[1]):
    #     max_ = max(data[:, j])
    #     min_ = min(data[:, j])
    #     if max_ == min_:
    #         continue
    #     for i in range(data.shape[0]):
    #         data[i][j] = (data[i][j] - min_) / (max_ - min_)
    idx = np.argwhere(label == 3).reshape(-1, )
    label = np.zeros((data.shape[0]),)
    label[idx] = 1
    point = data[idx]
    tree = KDTree(data)
    dis_sort, dis_idx = tree.query(data, k=kk+1)
    radius = np.mean(dis_sort[idx, 1:kk+1])
    print(radius)
    plotCluster(data, label, 'radius', idx, dis_idx, dis_sort, radius)

def plot_origin(file_path="../../data/aggSample/aggSample-6.txt"):
    file_path = './density.csv'
    data = pd.read_csv(file_path).values
    # data = np.loadtxt(file_path)
    # label = data[:, -1]
    # data = data[:, :-1]
    # for j in range(data.shape[1]):
    #     max_ = max(data[:, j])
    #     min_ = min(data[:, j])
    #     if max_ == min_:
    #         continue
    #     for i in range(data.shape[0]):
    #         data[i][j] = (data[i][j] - min_) / (max_ - min_)
    # idxs = np.argwhere(label==2).reshape(-1, )
    # data[idxs, 1] -= 0.2
    # data[idxs, 0] -= 0.05
    # tmp_data = np.concatenate([data, label.reshape(-1, 1)], axis=1)
    # np.savetxt('../../data/aggSample/aggSample-6.txt', tmp_data)
    label = np.zeros((data.shape[0], ))
    plotCluster(data, label, 'density-ori')

def plot_extend_neighbor(file_path="../../data/aggSample/aggSample-6.txt"):
    global kk
    data = np.loadtxt(file_path)
    # data = pd.read_csv(file_path).values
    label = data[:, -1]
    data = data[:, :-1]
    # for j in range(data.shape[1]):
    #     max_ = max(data[:, j])
    #     min_ = min(data[:, j])
    #     if max_ == min_:
    #         continue
    #     for i in range(data.shape[0]):
    #         data[i][j] = (data[i][j] - min_) / (max_ - min_)
    idx = np.argwhere(label == 3).reshape(-1, )
    tree = KDTree(data)
    dis_sort, dis_idx = tree.query(data, k=kk+1)
    radius = np.mean(dis_sort[:, 1:kk+1], axis=1)
    idx_list = []
    nn_list = []
    j = 1
    while radius[idx] >= dis_sort[idx, j]:
        idx_list.append(dis_idx[idx, j])
        nn_list.append(dis_idx[idx, j])
        t = 1
        while radius[dis_idx[idx, j]] >= dis_sort[dis_idx[idx, j], t]:
            idx_list.append(dis_idx[dis_idx[idx, j], t])
            t += 1
        j += 1
    idx_list = np.array(idx_list).reshape(-1, )
    nn_list = np.array(nn_list).reshape(-1, )
    print(idx_list)
    idx_set = set(idx_list.tolist())
    print(idx_set)
    label = np.zeros((data.shape[0], ))
    plotEN(data, label, 'extend-neighbor', idx, idx_set, nn_list, radius)

def plot_shrink(file_path="../../result-final/aggSample/ameliorated-16.txt"):
    data = np.loadtxt(file_path)
    label = data[:, -1]
    data = data[:, :-1]
    print(label)
    idx = np.argwhere(label == 3).reshape(-1, )
    label = np.zeros((data.shape[0], ))
    plotCluster(data, label, 'shrink', idx)


def plot_EN_diagraph(file_path="../../data/aggSample/aggSample-EN.csv"):
    global kk
    # data = np.loadtxt(file_path)
    data = pd.read_csv(file_path).values
    label = data[:, -1]
    data = data[:, :-1]
    idx = np.argwhere(label == 3).reshape(-1, )
    tree = KDTree(data)
    dis_sort, dis_idx = tree.query(data, k=kk+1)
    radius = np.mean(dis_sort[:, 1:kk+1], axis=1)
    # idx_list = []
    nn_list = []
    j = 1
    while radius[idx] >= dis_sort[idx, j]:
        # idx_list.append(dis_idx[idx, j])
        nn_list.append(dis_idx[idx, j])
        # t = 1
        # while radius[dis_idx[idx, j]] >= dis_sort[dis_idx[idx, j], t]:
        #     idx_list.append(dis_idx[dis_idx[idx, j], t])
        #     t += 1
        j += 1
    # idx_list = np.array(idx_list).reshape(-1, )
    nn_list = np.array(nn_list).reshape(-1, )
    # print(idx_list)
    # idx_set = set(idx_list.tolist())
    # print(idx_set)
    label = np.zeros((data.shape[0], ))
    plot_EN(data, label, 'extend-neighbor', idx, nn_list, radius, dis_sort, dis_idx)

# plot_origin()
plot_radius()
plot_extend_neighbor()
plot_shrink()
# plot_EN_diagraph()