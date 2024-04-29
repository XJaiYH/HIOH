import argparse
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# plot probability bar
def plot_valid_results_probability_bar(probabilities: np.ndarray, datasets_list: list, metrics_list: list, cls: str, args):
    x = np.arange(0, 10)
    width = 0.3
    datasets_list = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    for i, item in enumerate(metrics_list):
        plt.bar(x - width / 2, probabilities[:, i, 1], width, color='#4D96FF', label='invalid improvement')
        plt.bar(x + width / 2, probabilities[:, i, 0], width, color='#FF6363', label='valid improvement')
        plt.legend(loc='upper left', fontsize=18)
        plt.ylabel('Probability', size=28)
        plt.xticks(ticks=x, labels=datasets_list, size=20)
        plt.yticks(size=20)
        plt.xlabel("Datasets", size=28)
        plt.title(item, size=28)
        save_path = args.save_dir
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)
        plt.savefig(save_path + cls + "-probability-" + metrics_list[i] + ".png", bbox_inches='tight', dpi=300)
        plt.show()

# plot accuracy line

# plot mean value of valid and invalid results respectively
def plot_difference_compare2original(probabilities: np.ndarray, datasets_list: list, metrics_list: list, cls: str, args):
    '''
    绘制改善后的数据集相对聚类算法的提升度，即若提升能提升多少准确度，若下降能下降多少准确度
    :return:
    '''
    x = np.arange(0, 10)
    width = 0.3
    datasets_list = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, item in enumerate(metrics_list):
        plt.bar(x - width / 2, probabilities[:, i, 1], width, color='#4D96FF', label='invalid improvement')
        plt.bar(x + width / 2, probabilities[:, i, 0], width, color='#FF6363', label='valid improvement')
        plt.legend(loc='upper left', fontsize=18)
        plt.ylabel('Enhancement', size=28)
        plt.xticks(ticks=x, labels=datasets_list, size=20)
        plt.yticks(size=20)
        plt.xlabel("Datasets", size=28)
        plt.title(item, size=28)
        save_path = args.save_dir
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)
        plt.savefig(save_path + cls + "-Enhancement-" + metrics_list[i] + ".png", bbox_inches='tight', dpi=300)
        plt.show()

def highlight_max(x):
    is_max = x == x.max()
    return ['font-weight: bold' if val else ''
                for val in is_max]

def calculate_comparison_table(res_mean, clustering_method, datasets, metrics, save_path="../../table/original.xlsx"):
    res_mean = np.round(res_mean, 2)
    with pd.ExcelWriter(save_path) as writer:
        for ex_i in range(res_mean.shape[0]):
            sheetname = clustering_method[ex_i]
            res = res_mean[ex_i]
            # print(res.shape)
            res_df = pd.DataFrame(res, columns=metrics, index=datasets)
            # print(res_df)
            res_df = res_df.style.apply(highlight_max, axis=0)
            res_df.to_excel(writer, sheet_name=sheetname)

if __name__ == "__main__":
    # 该 phishing yeast balance
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../figure/robustness/', type=str, help='the director to save results')
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
    datasets = ['Letter', 'Archive', 'Hapt', 'Redwine', 'Bean'
                    , 'Wireless', 'Balance', 'Htru', 'Tae', 'Yeast']
    # datasets = ['whitewine']
    clustering_methods = ['kmeans', 'agg']
    metrics = ['ARI', 'NMI', 'FMI', 'PUR', 'VM']
    # record clustering accuracy of baseline clustering algorithm
    baseline = []
    for cls in clustering_methods:
        baseline_cls = []
        probabilities = []
        extent = []
        for item in datasets:
            res_path = args.dir + item + '/result-' + str(args.ratio) + '-clustering' + '-' + cls + '.csv'
            res = pd.read_csv(res_path).values
            baseline_res = np.mean(res, axis=0)
            baseline_cls.append(baseline_res)

        #     res_path = args.dir + item + '/result-' + str(args.ratio) + '-DM-averaging-fuzz' + '-' + cls + '.csv'
        #     res = pd.read_csv(res_path).values
        #     metric_probability = []
        #     metric_extent = []
        #     for i in range(len(metrics)):
        #         tmp_res = res[:, i].reshape(-1, )
        #         tmp_count = np.sum(tmp_res >= baseline_res[i])
        #         valid_percentage = tmp_count / tmp_res.shape[0]
        #         invalid_percentage = 1. - valid_percentage
        #         metric_probability.append([valid_percentage, invalid_percentage])
        #
        #         tmp_idx_higher = np.argwhere(tmp_res > baseline_res[i]).reshape(-1, )
        #         tmp_idx_lower = np.argwhere(tmp_res < baseline_res[i]).reshape(-1, )
        #         high_extent = (np.mean(tmp_res[tmp_idx_higher]) - baseline_res[i]) * valid_percentage
        #         low_extent = (np.mean(tmp_res[tmp_idx_lower]) - baseline_res[i]) * invalid_percentage
        #         metric_extent.append([high_extent, low_extent])
        #
        #     metric_probability = np.array(metric_probability)
        #     probabilities.append(metric_probability)
        #
        #     metric_extent = np.array(metric_extent)
        #     extent.append(metric_extent)
        #
        # probabilities = np.array(probabilities)
        # extent = np.array(extent)
        # print(probabilities.shape, extent.shape)

        baseline_cls = np.array(baseline_cls)
        baseline.append(baseline_cls)

        # plot_valid_results_probability_bar(probabilities, datasets, metrics, cls, args)
        # plot_difference_compare2original(extent, datasets, metrics, cls, args)
    baseline = np.array(baseline)[:, :, :len(metrics)]
    print(baseline.shape)
    calculate_comparison_table(baseline, clustering_methods, datasets, metrics)