import argparse
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# plot probability bar
def plot_valid_results_probability_bar(probabilities: np.ndarray, metrics_list: list, cls: str, args):
    global map_char, count
    x = np.arange(0, len(datasets_list))
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), constrained_layout=True)
    legend = True
    tick_set = [
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ]
    for i, item in enumerate(metrics_list):
        # plt.bar(x - width, probabilities[:, 0, i, 0], width, color='#FF6363', label='NAME', hatch='\\\\\\')
        # plt.bar(x, probabilities[:, 1, i, 0], width, color="#6DB9EF", label='HIBOG', hatch="///")
        # plt.bar(x + width, probabilities[:, 2, i, 0], width, color='#A7D397', label='HIAC', hatch="+++")
        # '#6DB9EF' '#A7D397'
        axs[0, i].plot(x, probabilities[0, :, 3, i, 0], color='#D875C7', label='SBCA', marker='s', linestyle='-.', markersize=10, linewidth=3)
        axs[0, i].plot(x, probabilities[0, :, 1, i, 0], color="#6499E9", label='HIBOG', linestyle='--', marker='o', markersize=10, linewidth=3)
        axs[0, i].plot(x, probabilities[0, :, 2, i, 0], color='#00DFA2', label='HIAC', linestyle='--', marker='^', markersize=10, linewidth=3)
        # axs[0, i].plot(x, probabilities[0, :, 3, i, 0], color='#CE49BF', label='Herd', linestyle='-.', marker='^',
        #                markersize=10, linewidth=3)

        axs[1, i].plot(x, probabilities[1, :, 3, i, 0], color='#D875C7', linestyle='-.', marker='s', markersize=10, linewidth=3)
        axs[1, i].plot(x, probabilities[1, :, 1, i, 0], color="#6499E9", marker='o', linestyle='--', markersize=10, linewidth=3)
        axs[1, i].plot(x, probabilities[1, :, 2, i, 0], color='#00DFA2', marker='^', linestyle='--', markersize=10, linewidth=3)
        # axs[1, i].plot(x, probabilities[1, :, 3, i, 0], color='#CE49BF', linestyle='-.', marker='^', markersize=10,
        #                linewidth=3)
        axs[0, i].axhline(y=0.85, c="#000000", ls="--", lw=4)
        axs[1, i].axhline(y=0.85, c="#000000", ls="--", lw=4)

        axs[0, i].plot(x, probabilities[0, :, 0, i, 0], color='#FF6363', label='HIOH', marker='*', markersize=15, linewidth=3)
        axs[1, i].plot(x, probabilities[1, :, 0, i, 0], color='#FF6363', marker='*', markersize=15, linewidth=3)
        axs[0, i].set_title(map_char[i+1], size=28, family='Times New Roman')
        axs[1, i].set_title(map_char[i+6], size=28, family='Times New Roman')
        axs[0, i].set_xticks(ticks=x, labels=datasets_list, size=20, family='Times New Roman', rotation=45)
        axs[1, i].set_xticks(ticks=x, labels=datasets_list, size=20, family='Times New Roman', rotation=45)
        axs[0, i].set_yticks(ticks=tick_set[i*2], size=25, family='Times New Roman')
        axs[1, i].set_yticks(ticks=tick_set[i*2+1], size=25, family='Times New Roman')

        if legend is True:
            fig.legend(loc='center', bbox_to_anchor=(0.51, 1.06), ncol=4, prop={'family': "Times New Roman", 'size': 28})
            legend = False
    axs[0, 0].set_ylabel('Probability', size=28, family='Times New Roman')
    axs[1, 0].set_ylabel('Probability', size=28, family='Times New Roman')
    for axe in axs:
        for ax in axe:
            # x1_label = ax.get_xticklabels()
            # [x1_label_temp.set_fontsize(15) for x1_label_temp in x1_label]
            y1_label = ax.get_yticklabels()
            [y1_label_temp.set_fontsize(25) for y1_label_temp in y1_label]
            [y1_label_temp.set_family('Times New Roman') for y1_label_temp in y1_label]
    save_path = args.save_dir
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    plt.savefig(save_path + "probability-test.png", bbox_inches='tight', dpi=400)
    plt.show()

# plot accuracy line

# plot mean value of valid and invalid results respectively
def plot_difference_compare2original(enhancement: np.ndarray, metrics_list: list, cls: str, args):
    '''
    绘制改善后的数据集相对聚类算法的提升度，即若提升能提升多少准确度，若下降能下降多少准确度
    :return:
    '''
    global map_char, count
    legend = True
    width = 0.15
    x = np.arange(0, len(datasets_list))
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(18, 20), constrained_layout=True)
    for i, item in enumerate(metrics_list):
        axs[i, 0].axhline(y=0, c="#B06161", ls="-", lw=3)
        axs[i, 1].axhline(y=0, c="#B06161", ls="-", lw=3)

        axs[i, 0].bar(x - 2*width, enhancement[0, :, 1, i, 0], width=width, color="#6DB9EF", label='HIBOG', hatch='///')
        axs[i, 0].bar(x - width, enhancement[0, :, 2, i, 0], width=width, color='#A7D397', label='HIAC', hatch='+++')
        axs[i, 0].bar(x + width, enhancement[0, :, 3, i, 0], width=width, color='#CE49BF', label='SBCA', hatch='---')
        axs[i, 0].bar(x + 2*width, enhancement[0, :, 0, i, 0], width=width, color='#FF6363', label='HIOH', hatch='\\\\\\')

        axs[i, 1].bar(x - 2*width, enhancement[1, :, 1, i, 0], width=width, color="#6DB9EF", hatch='///')
        axs[i, 1].bar(x - width, enhancement[1, :, 2, i, 0], width=width, color='#A7D397', hatch='+++')
        axs[i, 1].bar(x + width, enhancement[1, :, 3, i, 0], width=width, color='#CE49BF', hatch='\\\\\\')
        axs[i, 1].bar(x + 2*width, enhancement[1, :, 0, i, 0], width=width, color='#FF6363', hatch='\\\\\\')

        axs[i, 0].bar(x - 2*width, enhancement[0, :, 1, i, 1], width=width, color="#6DB9EF", hatch='///')
        axs[i, 0].bar(x - width, enhancement[0, :, 2, i, 1], width=width, color='#A7D397', hatch='+++')
        axs[i, 0].bar(x + width, enhancement[0, :, 3, i, 1], width=width, color='#CE49BF', hatch='---')
        axs[i, 0].bar(x + 2*width, enhancement[0, :, 0, i, 1], width=width, color='#FF6363', hatch='\\\\\\')

        axs[i, 1].bar(x - 2*width, enhancement[1, :, 1, i, 1], width=width, color="#6DB9EF", hatch='///')
        axs[i, 1].bar(x - width, enhancement[1, :, 2, i, 1], width=width, color='#A7D397', hatch='+++')
        axs[i, 1].bar(x + width, enhancement[1, :, 3, i, 1], width=width, color='#CE49BF', hatch='---')
        axs[i, 1].bar(x + 2*width, enhancement[1, :, 0, i, 1], width=width, color='#FF6363', hatch='\\\\\\')

        axs[i, 0].set_xticks(ticks=x, labels=datasets_list, size=20, rotation=0, family='Times New Roman')
        axs[i, 1].set_xticks(ticks=x, labels=datasets_list, size=20, rotation=0, family='Times New Roman')

        axs[i, 0].set_title(map_char[i + 1], size=25, family='Times New Roman')
        axs[i, 1].set_title(map_char[i + 6], size=25, family='Times New Roman')
        if legend is True:
            fig.legend(loc='center', bbox_to_anchor=(0.51, 1.03), ncol=3, prop={'family': "Times New Roman", 'size': 20})
            legend = False

    axs[0, 0].set_ylabel('Enhancement', size=25, family='Times New Roman')
    axs[1, 0].set_ylabel('Enhancement', size=25, family='Times New Roman')
    axs[2, 0].set_ylabel('Enhancement', size=25, family='Times New Roman')
    axs[3, 0].set_ylabel('Enhancement', size=25, family='Times New Roman')
    axs[4, 0].set_ylabel('Enhancement', size=25, family='Times New Roman')
    for axe in axs:
        for ax in axe:
            y_label = ax.get_yticklabels()
            [y_tmp.set_fontsize(20) for y_tmp in y_label]
            [y_tmp.set_family('Times New Roman') for y_tmp in y_label]
    save_path = args.save_dir
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    # plt.savefig(save_path + "Enhancement.png", bbox_inches='tight', dpi=400)
    plt.show()

def generateMap(map_char):
    map_char[1] = '(A)'
    map_char[2] = '(B)'
    map_char[3] = '(C)'
    map_char[4] = '(D)'
    map_char[5] = '(E)'
    map_char[6] = '(F)'
    map_char[7] = '(G)'
    map_char[8] = '(H)'
    map_char[9] = '(I)'
    map_char[10] = '(J)'
    map_char[11] = '(K)'
    map_char[12] = '(L)'
    map_char[13] = '(M)'
    map_char[14] = '(N)'

if __name__ == "__main__":
    # 该 phishing yeast balance
    global map_char, count
    map_char = {}
    count = 1
    generateMap(map_char)
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../figure/param_robustness_compare/', type=str, help='the director to save results')
    parser.add_argument('--dir', default='../../result-final/', type=str, help='the director of dataset')
    parser.add_argument('--itr', default=5, type=float, help='iteration times')
    parser.add_argument('--r_time', default=5, type=int, help='five times the average distance between all objects '
                                                              'in X and their closest neighbors')
    parser.add_argument('--data_name', default='bean.csv', type=str,
                        help='dataset name, one of {overlap1, overlap2, birch1, '
                             'birch2, iris, breast, iris, wine, htru, knowledge}')
    parser.add_argument('--NoEnhance', default=False, type=bool)
    parser.add_argument('--ratio', default=1, type=int)
    args = parser.parse_args()
    # datasets = ['tae', 'seeds', 'wireless', 'abalone', 'redwine', 'yeast', 'whitewine', 'waveform', 'balance', 'bean']
    datasets = ['Letter', 'Hapt', 'Redwine', 'Bean',
                   'archive', 'Wireless', 'balance', 'Htru', 'Tae', 'Yeast']
    # datasets = ['letter']
    clustering_methods = ['kmeans', 'agg']
    metrics = ['ARI', 'NMI', 'FMI', 'PUR', 'VM']
    ameliorate_methods = ['DM-averaging-fuzz-center-NLOGN-SKD', 'HIBOG', 'HIAC', 'SBCA']

    datasets_list = ['Let', 'Hap', 'Red', 'Bean', 'Arc', 'Wir', 'Bal', 'Htr',
                        'Tae', 'Yea']
    # ameliorate_methods = ['DM-averaging-fuzz-random-formal']
    # record clustering accuracy of baseline clustering algorithm
    tol_probability = []
    tol_extent = []
    for cls in clustering_methods:
        probabilities = []
        extent = []
        for item in datasets:
            res_path = args.dir + item + '/result-' + str(args.ratio) + '-clustering' + '-' + cls + '.csv'
            res = pd.read_csv(res_path).values
            baseline_res = np.mean(res, axis=0)
            # baseline_res = np.round(baseline_res, 2)
            print(baseline_res)
            algo_probability = []
            algo_extent = []
            for algo in ameliorate_methods:
                res_path = args.dir + item + '/result-' + str(args.ratio) + '-' + algo + '-' + cls + '.csv'
                res = pd.read_csv(res_path).values
                if algo == 'DM-averaging-fuzz-center-NLOGN-SKD':
                    res = res[11:61]
                    print(res.shape)
                metric_probability = []
                metric_extent = []
                for i in range(len(metrics)):
                    if i != -1:
                        tmp_res = res[:, i]
                        # tmp_res = np.round(res[:, i].reshape(-1, ), 2)
                        tmp_count = np.sum(tmp_res > baseline_res[i])
                        valid_percentage = tmp_count / tmp_res.shape[0]
                        invalid_percentage = 1. - valid_percentage
                        metric_probability.append([valid_percentage, invalid_percentage])

                        tmp_idx_higher = np.argwhere(tmp_res > baseline_res[i]).reshape(-1, )
                        tmp_idx_lower = np.argwhere(tmp_res <= baseline_res[i]).reshape(-1, )
                        high_extent = (np.mean(tmp_res[tmp_idx_higher]) - baseline_res[i]) * valid_percentage
                        low_extent = (np.mean(tmp_res[tmp_idx_lower]) - baseline_res[i]) * invalid_percentage
                        metric_extent.append([high_extent, low_extent])

                metric_probability = np.array(metric_probability)
                algo_probability.append(metric_probability)

                metric_extent = np.array(metric_extent)
                algo_extent.append(metric_extent)

            algo_probability = np.array(algo_probability)
            probabilities.append(algo_probability)

            algo_extent = np.array(algo_extent)
            extent.append(algo_extent)
        probabilities = np.array(probabilities)
        extent = np.array(extent)
        tol_probability.append(probabilities)
        tol_extent.append(extent)
    tol_probability = np.array(tol_probability)
    tol_extent = np.array(tol_extent)
    print(tol_probability.shape, tol_extent.shape)

    metrics = ['ARI', 'NMI', 'FMI', 'PUR', 'VM']
    plot_valid_results_probability_bar(tol_probability, metrics, cls, args)
    # plot_difference_compare2original(tol_extent, metrics, cls, args)
