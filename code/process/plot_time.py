import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def autolabel(rects, height):
    for i, rect in enumerate(rects):
        value = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.08, 1.03*value, '%s' % int(height[i]), size=15, family="Times new roman")

def getCoords(coords: np.ndarray, label, y):
    label = np.array(label).reshape(-1, )
    y = np.array(y).reshape(-1, )
    coords = np.array(coords)
    # print(coords)
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            for t in range(len(label)):
                if label[t] < coords[i, j]:
                    continue
                else:
                    prop = (coords[i, j] - label[t - 1]) / (label[t] - label[t - 1])
                    coords[i, j] = y[t - 1] + (y[t] - y[t - 1]) * prop
                    break
    return coords

def plot_time(res, dataset, optimal):
    data_y_label = np.array([0, 0.1, 1, 10, 100, 500, 5000, 50000, 60000])
    data_y = np.array([0, 1, 2, 3, 4, 5, 7, 9, 10])
    coords = res
    tmp = [[0.0037431716918945312, 0.38530421257019043, 0.2941467761993408, 0.9612798690795898, 0.25112223625183105, 32.74704027175903]
        ,[0.07556581497192383, 1.0753076076507568, 2.594766139984131, 4.57983136177063, 7.343584775924683, 133.54106974601746]]
    # coords = coords.T
    tmp = np.array(tmp)
    coords[1, 0:6] = tmp[0]
    coords[2, 0:6] = tmp[1]
    res = getCoords(coords, label=data_y_label, y=data_y)

    gap = 0.22
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(16, 4))
    cm1 = plt.bar(x - 3 / 2 * gap, res[0, :], width=gap, color='#FF6363', label='HIOH', hatch="///")
    # autolabel(cm1, optimal[0])
    cm2 = plt.bar(x - 1 / 2 * gap, res[1, :], width=gap, color='#6DB9EF', label='HIBOG', hatch="\\\\\\")
    # autolabel(cm2, optimal[1])
    cm3 = plt.bar(x + 1 / 2 * gap, res[2, :], width=gap, color='#A7D397', label='HIAC', hatch="+++")
    # autolabel(cm3, optimal[2])
    cm4 = plt.bar(x + 3 / 2 * gap, res[3, :], width=gap, color='#BB9CC0', label='SBCA', hatch="...")
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=4, prop={'family':"Times New Roman", 'size': 20})
    plt.ylabel('Runtime(s)', family="Times New Roman", size=20)
    plt.xticks(ticks=x, labels=dataset, family="Times New Roman", size=20, rotation=0)
    plt.yticks(ticks=data_y, labels=data_y_label, family="Times New Roman", size=20)
    plt.xlabel("Dataset", family="Times New Roman", size=20)
    #plt.title("Runtime Comparison", size=20)
    save_path = args.save_fig
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    plt.savefig(save_path + "runtime.png", bbox_inches='tight', dpi=300)
    plt.show()

def sub_NLogN_complexity(dataset):
    colors = ['#4363d8', '#95CD41', '#125B50', '#f032e6',
              '#fabed4', '#469990',  '#9A6324',
              '#125B50', '#4D96FF', '#FFD93D', '#FF6363', '#C36A2D', '#22577E', '#4700D8', '#F900DF',
              '#95CD41',
              '#FF5F00', '#40DFEF', '#8E3200', '#001E6C', '#B91646']
    tol_res = []
    k_set = [15, 25, 35, 45, 55, 65]
    # k_set = [10, 20, 30, 40, 50]
    sub_dataset = dataset
    for name in sub_dataset:
        path = '../../result-time/' + name + '-iter-N-'
        ori_path = '../../data/' + name + '/' + name + '-1.csv'
        for ii in k_set:
            data = pd.read_csv(ori_path).values
            print(name, data.shape, len(set(data[:, -1].tolist())))
            tmp_path = path + str(ii) + '.txt'
            res = np.loadtxt(tmp_path).reshape(-1, ).astype(int)
            res = np.concatenate([[data.shape[0]], res], axis=0)
            tol_res.append(res)
    # tol_res = np.array(tol_res)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), constrained_layout=True)
    # plt.subplots_adjust(hspace=0.5, wspace=0.2)
    legend = True
    mks = 8
    tick_set = [
        [0., 30., 60., 90., 120, 150],
        [0., 300., 600., 900., 1200],
        [0., 150., 300., 450., 600],
        [0., 500., 1000., 1500., 2000],
        [0., 400., 800., 1200., 1600],
        [0., 3000., 6000., 9000., 12000],
        [0., 3000., 6000., 9000., 12000, 15000],
        [0., 500., 1000., 1500., 2000],
        [0., 4000., 8000., 12000., 16000],
        [0., 2000., 4000., 6000., 8000, 10000]
    ]
    for i in range(10):
        if i == 0:
            xi = np.arange(tol_res[i*6].shape[0])
            axes[i//5, i%5].plot(xi, tol_res[i*6], c=colors[0], linewidth=3, ls='--', marker='X', markersize=mks, label='K=15')
            xi = np.arange(tol_res[i * 6 + 1].shape[0])
            axes[i//5, i%5].plot(xi, tol_res[i*6+1], c=colors[1], linewidth=3, ls='-.', marker='h', markersize=mks, label='K=25')
            xi = np.arange(tol_res[i * 6 + 2].shape[0])
            axes[i//5, i%5].plot(xi, tol_res[i*6+2], c=colors[3], linewidth=3, ls='--', marker='s', markersize=mks, label='K=35')
            xi = np.arange(tol_res[i * 6 + 3].shape[0])
            axes[i//5, i%5].plot(xi, tol_res[i*6+3], c=colors[2], linewidth=3, ls='-.', marker='v', markersize=mks, label='K=45')
            xi = np.arange(tol_res[i * 6 + 4].shape[0])
            axes[i//5, i%5].plot(xi, tol_res[i*6+4], c='#C69774', linewidth=3, ls='--', marker='^', markersize=mks, label='K=55')
            xi = np.arange(tol_res[i * 6 + 5].shape[0])
            axes[i//5, i%5].plot(xi, tol_res[i*6+5], c='#FF6363', linewidth=3, ls='-', marker='o', markersize=mks, label='K=65')
        else:
            xi = np.arange(tol_res[i * 6].shape[0])
            axes[i // 5, i % 5].plot(xi, tol_res[i * 6], c=colors[0], linewidth=3, ls='--', marker='X', markersize=mks)
            xi = np.arange(tol_res[i * 6 + 1].shape[0])
            axes[i // 5, i % 5].plot(xi, tol_res[i * 6 + 1], c=colors[1], linewidth=3, ls='-.', marker='h', markersize=mks)
            xi = np.arange(tol_res[i * 6 + 2].shape[0])
            axes[i // 5, i % 5].plot(xi, tol_res[i * 6 + 2], c=colors[3], linewidth=3, ls='--', marker='s', markersize=mks)
            xi = np.arange(tol_res[i * 6 + 3].shape[0])
            axes[i // 5, i % 5].plot(xi, tol_res[i * 6 + 3], c=colors[2], linewidth=3, ls='-.', marker='v', markersize=mks)
            xi = np.arange(tol_res[i * 6 + 4].shape[0])
            axes[i // 5, i % 5].plot(xi, tol_res[i * 6 + 4], c='#C69774', linewidth=3, ls='--', marker='^', markersize=mks)
            xi = np.arange(tol_res[i * 6 + 5].shape[0])
            axes[i//5, i%5].plot(xi, tol_res[i*6+5], c='#FF6363', linewidth=3, ls='-', marker='o', markersize=mks)
        axes[i//5, i%5].xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        axes[i//5, i%5].set_title(sub_dataset[i], size=25, family="Times New Roman")
        # if i >= 5:
        axes[i//5, i%5].set_xlabel('Iteration', size=25, family="Times New Roman")
        axes[i//5, i%5].set_yticks(ticks=tick_set[i], size=25, family='Times New Roman')
        axes[i//5, i%5].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.legend(loc='center', bbox_to_anchor=(0.51, 1.05), ncol=6, prop={'family': "Times New Roman", 'size': 28})
    axes[0, 0].set_ylabel('N(Size)', size=30, family="Times New Roman")
    axes[1, 0].set_ylabel('N(Size)', size=30, family="Times New Roman")

    # x1 = np.arange(tol_res[1].shape[0])
    # axes[1].plot(x1, tol_res[1], c='#FF6363', linewidth=2, marker='.', markersize=10)
    # axes[1].xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    # axes[1].set_title(sub_dataset[1], size=20)
    # axes[1].set_xlabel('Iteration', size=20)
    # axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    #
    # x2 = np.arange(tol_res[2].shape[0])
    # axes[2].plot(x2, tol_res[2], c='#FF6363', linewidth=2, marker='.', markersize=10)
    # axes[2].xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    # axes[2].set_title(sub_dataset[2], size=20)
    # axes[2].set_xlabel('Iteration', size=20)
    # axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    #
    # x3 = np.arange(tol_res[3].shape[0])
    # axes[3].plot(x3, tol_res[3], c='#FF6363', linewidth=2, marker='.', markersize=10)
    # axes[3].xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    # axes[3].set_title(sub_dataset[3], size=20)
    # axes[3].set_xlabel('Iteration', size=20)
    # axes[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    #
    # x4 = np.arange(tol_res[4].shape[0])
    # axes[4].plot(x4, tol_res[4], c='#FF6363', linewidth=2, marker='.', markersize=10)
    # axes[4].xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    # axes[4].set_title(sub_dataset[4], size=20)
    # axes[4].set_xlabel('Iteration', size=20)
    # axes[4].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    #
    # x5 = np.arange(tol_res[5].shape[0])
    # axes[5].plot(x5, tol_res[5], c='#FF6363', linewidth=2, marker='.', markersize=10)
    # axes[5].xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    # axes[5].set_title(sub_dataset[5], size=20)
    # axes[5].set_xlabel('Iteration', size=20)
    # axes[5].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for axs in axes:
        for ax in axs:
            x1_label = ax.get_xticklabels()
            [x1_label_temp.set_fontsize(22) for x1_label_temp in x1_label]
            y1_label = ax.get_yticklabels()
            [y1_label_temp.set_fontsize(22) for y1_label_temp in y1_label]

    plt.savefig('../../figure/time/iterCompareN-RangeK.png', bbox_inches='tight', dpi=400)
    plt.show()


def plot_time_comparison(args, boost_method, dataset):
    time_all = []
    optimal = []
    for method in boost_method:
        time_dataset = []
        time_optimal = []
        for name in dataset:
            path = args.dir + name + '/' + 'result-' + str(args.ratio) + '-' + method + '-kmeans.csv'
            path2 = args.dir + name + '/' + 'result-' + str(args.ratio) + '-' + method + '-agg.csv'

            res1 = pd.read_csv(path).values[:, :8]
            res2 = pd.read_csv(path2).values[:, :8]
            if method == 'DM-averaging-fuzz-center-NLOGN-SKD':
                res1 = res1[10:]
                res2 = res2[10:]
            length = min(res1.shape[0], res2.shape[0])
            res1 = res1[:length]
            res2 = res2[:length]
            res = res1 + res2
            max_ = np.max(res[:, 1])
            idx = np.argwhere(res[:, 1] == max_).reshape(-1, )
            # max1 = np.max(res1[:, 1])
            # idx1 = np.argwhere(res1[:, 1] == max1).reshape(-1, )
            # max2 = np.max(res1[:, 1])
            # idx2 = np.argwhere(res1[:, 1] == max2).reshape(-1, )
            # res1 = res1[idx1]
            # res2 = res2[idx2]
            ress = res1[idx]

            time = np.min(ress[:, 7])
            optimal_idx = np.argmin(ress[:, 7])
            time_optimal.append(idx[optimal_idx])
            print('best idx in 【' + method + '】【' + name + '】: ' + str(idx[optimal_idx]))
            final_time = time

            time_dataset.append(final_time)
        time_dataset = np.array(time_dataset)
        time_all.append(time_dataset)
        time_optimal = np.array(time_optimal)
        optimal.append(time_optimal)
    time_all = np.array(time_all)  # len(boost_method) * len(dataset)
    # print(time_all)
    optimal = np.array(optimal)
    np.savetxt("../../result-time/or_time.txt", optimal, fmt='%d')
    optimal[0, :] = optimal[0, :] + 5
    optimal[1, :] = optimal[1, :] % 10 + 1
    optimal[2, :] = (optimal[2, :] % 10 + 1) * 4
    np.savetxt("../../result-time/time.txt", optimal, fmt='%d')
    plot_time(time_all, dataset, optimal)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fig', default='../../figure/time/', type=str, help='the director to save results')
    parser.add_argument('--dir', default='../../result-final/', type=str, help='the director of dataset')
    parser.add_argument('--itr', default=5, type=float, help='iteration times')
    parser.add_argument('--r_time', default=5, type=int, help='five times the average distance between all objects '
                                                              'in X and their closest neighbors')
    parser.add_argument('--ratio', default=1, type=int)
    parser.add_argument('--has_label', default=True, type=bool)
    parser.add_argument('--NoEnhance', default=False, type=bool)
    parser.add_argument('--k', default=10, type=int)
    args = parser.parse_args()
    if os.path.isdir(args.save_fig) is False:
        os.mkdir(args.save_fig)

    dataset = ['Tae', 'Yeast',  'Balance', 'Wireless', 'Redwine', 'Bean', 'Letter', 'HAPT'
                , 'Htru', 'Archive']
    boost_method = ['DM-averaging-fuzz-center-NLOGN-SKD', 'HIBOG', 'HIAC', 'SBCA']
    # plot_time_comparison(args, boost_method, dataset)
    sub_NLogN_complexity(dataset)



