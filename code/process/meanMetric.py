import argparse
import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
global plot_data_name
plot_data_name = ['Arc', 'Win', 'Red', 'Bea', 'Whi', 'Arc', 'Wir', 'Bal', 'Tae', 'See', 'Aba', 'Yea', 'Let']

def plotMetrics(score: np.ndarray, dataname: str, method_list: list, save_path: str, metric: list, cls_name, func:str):
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324',
              '#125B50', '#4D96FF', '#FFD93D', '#FF6363','#CE49BF', '#C36A2D',  '#22577E', '#4700D8', '#F900DF', '#95CD41',
             '#FF5F00', '#40DFEF', '#8E3200', '#001E6C',  '#B91646']
    for j in range(len(metric)):
        fig = plt.subplots(figsize=(10, 5))
        for i in range(len(method_list)):
            if method_list[i] == 'DM':
                plt.plot(ratios, score[i, :, j], color=colors[i], label=method_list[i], linewidth=3, marker='*', markersize=15)
            else:
                plt.plot(ratios, score[i, :, j], color=colors[i], label=method_list[i], linewidth=3)
        plt.legend(loc='upper left', fontsize=18)
        plt.ylabel(metric[j], size=28)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel("density rate", size=28)
        plt.title(dataname + "-" + cls_name + "-" + func + " :" + metric[j], size=28)
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        plt.savefig(save_path + func + "_" + cls + "_" + metric[j] + ".png", bbox_inches='tight', dpi=300)
        plt.show()

def plotMetricsSingleRate(score: np.ndarray, method_list: list, save_path: str, metric: list, cls_list: list, ty):
    global map_char, count, plot_data_name
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324',
              '#125B50', '#4D96FF', '#FFD93D', '#FF6363','#CE49BF', '#C36A2D',  '#22577E', '#4700D8', '#F900DF', '#95CD41',
             '#FF5F00', '#40DFEF', '#8E3200', '#001E6C',  '#B91646']
    x = [i for i in range(len(plot_data_name))]
    y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(score.shape)
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(22, 7), constrained_layout=True)
    # fig.subplots_adjust(hspace=0.35, wspace=0.3)
    legend = 0
    for t in range(len(cls_list)):
        for j in range(len(metric)):
            for i in range(len(method_list)):
                if method_list[i] == 'HOP':
                    axs[t, j].plot(x, score[:, t, i, j].reshape(-1, ), color=colors[i], label=method_list[i], linewidth=3, marker='*', markersize=15)
                else:
                    axs[t, j].plot(x, score[:, t, i, j].reshape(-1, ), color=colors[i], label=method_list[i], linewidth=3, linestyle='--', marker='o', markersize=8)
                axs[t, j].set_ylabel(metric[j], size=15, family='Times New Roman')
                axs[t, j].set_xticks(x, plot_data_name, size=15, rotation=45, family='Times New Roman')
                axs[t, j].set_yticks(y, y, size=15, family='Times New Roman')
                axs[t, j].set_ylim(round(min(score[:, t, i, j])-0.05, 1), 1.05)
                axs[t, j].set_title(map_char[t*5+j+1], size=20, family='Times New Roman')
                legend += 1
                if legend == len(method_list):
                    fig.legend(loc='center', bbox_to_anchor=(0.5, 1.0), ncol=len(method_list), prop={'family': "Times New Roman", 'size': 20})
    for axe in axs:
        for ax in axe:
            x1_label = ax.get_xticklabels()
            [x1_label_temp.set_fontsize(15) for x1_label_temp in x1_label]
            y1_label = ax.get_yticklabels()
            [y1_label_temp.set_fontsize(15) for y1_label_temp in y1_label]
            [y1_label_temp.set_family('Times New Roman') for y1_label_temp in y1_label]
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    # plt.savefig(save_path + "absolution-test-beta-" + ty + ".png", bbox_inches='tight', dpi=300)
    plt.show()

def plotAbsoluteDensityMeanGraph(score: np.ndarray, method_list: list, save_path: str, metric: list, cls_list: list, func:str, titile:str):
    global map_char, count, plot_data_name
    x = np.arange(len(plot_data_name))
    y = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(score.shape)
    score = score.transpose(1, 2, 0, 3)
    score = np.mean(score, axis=-1)
    print(score.shape)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5), constrained_layout=True)
    # fig.subplots_adjust(hspace=0.35, wspace=0.3)
    legend = 0
    width = 0.32
    for t in range(len(cls_list)):
        axs[t].bar(x-width/2, score[t, 0, :], width, color='#FF6363', hatch='///', label=method_list[0])
        axs[t].bar(x+width/2, score[t, 1, :], width, color='#6DB9EF', hatch='\\\\\\', label=method_list[1])

        axs[t].set_xticks(x, plot_data_name, size=28, rotation=0, family='Times New Roman')
        axs[t].set_yticks(y, y, size=28, family='Times New Roman')
        axs[t].set_ylim(0.25, 1.0)
        axs[t].set_title(map_char[t + 1], size=30, family='Times New Roman')
        if legend == 0:
            legend += 1
            fig.legend(loc='center', bbox_to_anchor=(0.52, 1.1), ncol=2, prop={'family': "Times New Roman", 'size': 28})
    axs[0].set_ylabel('Mean Accuracy', size=30, family='Times New Roman')
    for ax in axs:
        x1_label = ax.get_xticklabels()
        [x1_label_temp.set_fontsize(28) for x1_label_temp in x1_label]
        y1_label = ax.get_yticklabels()
        [y1_label_temp.set_fontsize(28) for y1_label_temp in y1_label]
        [y1_label_temp.set_family('Times New Roman') for y1_label_temp in y1_label]
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    plt.savefig(save_path + "absolution-density-mean.png", bbox_inches='tight', dpi=400)
    plt.show()

def highlight_max(x):
    is_max = x == x.max()
    return ['font-weight: bold' if val else ''
                for val in is_max]

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

def plot_absolution_hopping_mean(res, metrics, cls, args):
    x = np.arange(5)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
    width = 0.13
    for i, item in enumerate(cls):
        axs[i].bar(x - width*2.5, res[i, 0], width=width, color='#FF90BC', label='HIOH-F', hatch='\\\\\\')
        axs[i].bar(x - width*1.5, res[i, 1], width=width, color='#A7D397', label='HIOH-N', hatch='+++')
        axs[i].bar(x - width*0.5, res[i, 2], width=width, color='#6DB9EF', label='HIOH-L', hatch='...')
        axs[i].bar(x + width*0.5, res[i, 3], width=width, color='#D1BB9E', label='HIOH-H', hatch='\/\/\/')
        axs[i].bar(x + width*1.5, res[i, 4], width=width, color='#BB9CC0', label='HIOH-R', hatch='---')
        axs[i].bar(x + width*2.5, res[i, 5], width=width, color='#FF6363', label='HIOH', hatch='///')
        # axs[i].plot(x, res[i, 1], color='#FF90BC', label='NAME-farthest', linewidth=3, marker='o', markersize=7)
        # axs[i].plot(x, res[i, 2], color='#A7D397', label='NAME-nearest', linewidth=3, marker='o', markersize=7)
        # axs[i].plot(x, res[i, 3], color='#6DB9EF', label='NAME-lowest', linewidth=3, marker='o', markersize=7)
        # axs[i].plot(x, res[i, 4], color='#1B4242', label='NAME-Highest', linewidth=3, marker='o', markersize=7)
        # axs[i].plot(x, res[i, 5], color='#BB9CC0', label='NAME-random', linewidth=3, marker='o', markersize=7)
        # axs[i].plot(x, res[i, 0], color='#FF6363', label='NAME', linewidth=3, marker='*', markersize=12)
    axs[0].set_ylim((0.6, 0.82))
    axs[1].set_ylim((0.6, 0.82))
    axs[0].legend(loc='center', bbox_to_anchor=(1.01, 1.2), ncol=6, prop={'family': "Times New Roman", 'size': 20.5})
    axs[0].set_ylabel('Mean', size=30, family='Times New Roman')
    axs[0].set_xticks(ticks=x, labels=metrics, size=28, family='Times New Roman')
    axs[1].set_xticks(ticks=x, labels=metrics, size=28, family='Times New Roman')
    axs[0].set_yticks(ticks=[0.6, 0.65, 0.7, 0.75, 0.8], size=28, family='Times New Roman')
    axs[1].set_yticks(ticks=[0.6, 0.65, 0.7, 0.75, 0.8], size=28, family='Times New Roman')
    plt.yticks(size=20, family='Times New Roman')
    axs[0].set_xlabel('Metric', size=30, family='Times New Roman')
    axs[1].set_xlabel('Metric', size=30, family='Times New Roman')
    axs[0].set_title('(A)', size=30, family='Times New Roman')
    axs[1].set_title('(B)', size=30, family='Times New Roman')
    for ax in axs:
        x1_label = ax.get_xticklabels()
        [x1_label_temp.set_fontsize(28) for x1_label_temp in x1_label]
        [x1_label_temp.set_family('Times New Roman') for x1_label_temp in x1_label]
        y1_label = ax.get_yticklabels()
        [y1_label_temp.set_fontsize(28) for y1_label_temp in y1_label]
        [y1_label_temp.set_family('Times New Roman') for y1_label_temp in y1_label]
    args.save_fig = '../../figure/absolution-hopping/'
    save_path = args.save_fig
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    plt.savefig(save_path + "absolution-hopping-mean.png", bbox_inches='tight', dpi=300)
    plt.show()

def calculate_meanvalue_for_absolution_hopping(res_mean, metrics:list, clustering_method:list, args,
                                               save_path="../../table/absolution-hopping-meanvalue.xlsx"):
    res_mean = res_mean.transpose(1, 2, 3, 0)
    res_mean = np.mean(res_mean, axis=-1)
    print(res_mean.shape)
    with pd.ExcelWriter(save_path) as writer:
        for ex_i in range(res_mean.shape[0]):
            sheetname = clustering_method[ex_i]
            res = res_mean[ex_i]
            res_df = pd.DataFrame(res, columns=metrics, index=boost_method)
            res_df = res_df.round(3)
            res_df = res_df.style.apply(highlight_max, axis=0)
            res_df.to_excel(writer, sheet_name=sheetname)
    plot_absolution_hopping_mean(res_mean, metrics, clustering_method, args)

def calculate_comparison_table(res_mean, boost_method, save_path="../../table/comparison-mean-percent.xlsx"):
    print(res_mean.shape)
    result = res_mean.transpose(2, 0, 1, 3)
    print(result.shape)
    result = result.reshape(5, 10, -1)
    print(result.shape)
    result = np.round(result, 2)
    flag_arr = np.zeros((6, 11, 10), dtype=int)
    origin_acc = [[0.13,0.36,0.17,0.3,0.36,0.14,0.4,0.18,0.33,0.41],
                [0.45,0.59,0.55,0.6,0.59,0.49,0.63,0.59,0.66,0.63],
                [0.45,0.58,0.55,0.58,0.58,0.46,0.59,0.55,0.59,0.59],
                [0.23,0.37,0.44,0.73,0.37,0.3,0.46,0.5,0.82,0.46],
                [0.33,0.53,0.54,0.5,0.54,0.31,0.53,0.53,0.51,0.53],
                [0.47,0.64,0.64,0.68,0.64,0.57,0.73,0.72,0.71,0.73],
                [0.63,0.71,0.79,0.92,0.71,0.64,0.72,0.79,0.92,0.72],
                [0.81,0.69,0.97,0.98,0.69,0.54,0.43,0.94,0.95,0.43],
                [0.44,0.64,0.68,0.67,0.65,0.47,0.66,0.71,0.67,0.66],
                [0.62,0.78,0.78,0.75,0.78,0.68,0.8,0.8,0.81,0.8]]
    origin_acc = np.array(origin_acc)
    origin_acc = np.round(origin_acc, 2)


    datasets = ['&\emph{Letter}', '&\emph{Archive}', '&\emph{HPAT}', '&\emph{Redwine}', '&\emph{Bean}', '&\emph{Wireless}',
                '&\emph{Balance}', '&\emph{Htru}', '&\emph{Tae}', '&\emph{Yeast}', '&\\textbf{Average}']
    for i in range(result.shape[1]): # for each dataset
        for j in range(result.shape[2]): # for each measure
            tmp = result[:, i, j]
            max_ = np.max(tmp)
            for t in range(3):
                if result[t, i, j] == max_:
                    flag_arr[t, i, j] = 1
                if result[t, i, j]  < origin_acc[i, j]:
                    flag_arr[t, i, j] = -1
    flag_arr[0, -1, :] = 1
    flag_arr[1, -1, :] = -1
    flag_arr[2, -1, :] = -1

    column = metrics * 2
    ori_mean = np.mean(origin_acc, axis=0)
    ori_mean = np.round(ori_mean, 2)

    with pd.ExcelWriter(save_path) as writer:
        for ex_i in range(result.shape[0]):
            sname = boost_method[ex_i]
            res = result[ex_i]
            res_mean = np.mean(res, axis=0)
            res_mean = np.round(res_mean, 2)

            mean_percent = (res_mean - ori_mean) / ori_mean * 100
            mean_percent = mean_percent.astype(np.int)
            percent = (res - origin_acc) / origin_acc * 100
            percent = percent.astype(np.int)
            res = np.concatenate([res, res_mean.reshape(1, -1)], axis=0)
            percent = np.concatenate([percent, mean_percent.reshape(1, -1)], axis=0)

            res_df = pd.DataFrame(res, columns=column, index=datasets)
            print(res.shape)
            # res_df = res_df.round(3)
            for i in range(res_df.shape[0]):
                for j in range(res_df.shape[1]):
                    if percent[i, j] > 0:
                        prefix = '\\uparrow'
                    elif percent[i, j] < 0:
                        prefix = '\\downarrow'
                    elif percent[i,j] == 0:
                        prefix = ''
                    percent[i, j] = abs(percent[i, j])
                    res_df.iloc[i, j] = f'{res_df.iloc[i, j]}$^' + '{' + prefix + str(percent[i, j]) + '\\%}$'
                    if flag_arr[ex_i, i, j] == 1:
                        res_df.iloc[i, j] = f"\\textbf{{{res_df.iloc[i, j]}}}"
                    # elif flag_arr[ex_i, i, j] == -1:
                    #     res_df.iloc[i, j] = f"{res_df.iloc[i, j]}$^" + '{' + '\\boldmath{\\downarrow}}$'
            res_df.iloc[:, :] = res_df.iloc[:, :].applymap(lambda x: '&{}'.format(x))
            res_df.iloc[:, -1] = res_df.iloc[:, -1].apply(lambda x: '{}\\\\'.format(x))
            res_df.to_excel(writer, sheet_name=sname)

    # calculate Mean Enhancement
    # save_path = '../../table/MeanEnhancePercent.xlsx'
    # with pd.ExcelWriter(save_path) as writer:
    #     final_res = []
    #     average_res = []
    #     for ex_i in range(result.shape[0]):
    #         sname = 'MeanEnhancePercent'
    #         res = result[ex_i]
    #         print(res.shape)
    #         res = np.round(res, 3)
    #         res = np.mean(res, axis=0)
    #         average_res.append(res)
    #         res = (res - ori_mean) / ori_mean * 100
    #         res = np.round(res, 1)
    #         final_res.append(res)
    #     final_res = np.array(final_res)
    #     average_res = np.array(average_res)
    #     average_res = np.round(average_res, 3)
    #     final_res = np.concatenate([final_res, average_res], axis=0)
    #     res_df = pd.DataFrame(final_res, columns=column, index=method*2)
    #     for i in [0, 3]:
    #         for j in range(10):
    #             res_df.iloc[i, j] = f"\\textbf{{{res_df.iloc[i, j]}}}"
    #     res_df.iloc[:, :] = res_df.iloc[:, :].applymap(lambda x: '&{}'.format(x))
    #     res_df.iloc[:3, :] = res_df.iloc[:3, :].applymap(lambda x: '{}\%'.format(x))
    #     res_df.iloc[:, -1] = res_df.iloc[:, -1].apply(lambda x: '{}\\\\'.format(x))
    #     res_df.to_excel(writer, sheet_name=sname)

def calculate_comparison_table_maxima(res_max, boost_method: list, save_path="../../table/comparison-max-percent.xlsx"):
    print(res_max.shape)
    print(res_max.shape)
    result = res_max.transpose(2, 0, 1, 3)
    print(result.shape)
    result = result.reshape(3, 10, -1)
    print(result.shape)
    result = np.round(result, 2)
    flag_arr = np.zeros((3, 11, 10), dtype=int)
    origin_acc = [[0.13,0.36,0.17,0.3,0.36,0.14,0.4,0.18,0.33,0.41],
                [0.45,0.59,0.55,0.6,0.59,0.49,0.63,0.59,0.66,0.63],
                [0.45,0.58,0.55,0.58,0.58,0.46,0.59,0.55,0.59,0.59],
                [0.23,0.37,0.44,0.73,0.37,0.3,0.46,0.5,0.82,0.46],
                [0.33,0.53,0.54,0.5,0.54,0.31,0.53,0.53,0.51,0.53],
                [0.47,0.64,0.64,0.68,0.64,0.57,0.73,0.72,0.71,0.73],
                [0.63,0.71,0.79,0.92,0.71,0.64,0.72,0.79,0.92,0.72],
                [0.81,0.69,0.97,0.98,0.69,0.54,0.43,0.94,0.95,0.43],
                [0.44,0.64,0.68,0.67,0.65,0.47,0.66,0.71,0.67,0.66],
                [0.62,0.78,0.78,0.75,0.78,0.68,0.8,0.8,0.81,0.8]]
    origin_acc = np.array(origin_acc)
    origin_acc = np.round(origin_acc, 2)

    datasets = ['&\emph{Letter}', '&\emph{Archive}', '&\emph{HPAT}', '&\emph{Redwine}', '&\emph{Bean}', '&\emph{Wireless}',
                '&\emph{Balance}', '&\emph{Htru}', '&\emph{Tae}', '&\emph{Yeast}', '&\\textbf{Average}']
    for i in range(result.shape[1]):  # for each dataset
        for j in range(result.shape[2]):  # for each measure
            tmp = result[:, i, j]
            max_ = np.max(tmp)
            for t in range(3):
                if result[t, i, j] == max_:
                    flag_arr[t, i, j] = 1
                if result[t, i, j] < origin_acc[i, j]:
                    flag_arr[t, i, j] = -1
    flag_arr[0, -1, :] = 1
    flag_arr[1, -1, :] = -1
    flag_arr[2, -1, :] = -1

    column = metrics * 2
    ori_mean = np.mean(origin_acc, axis=0)
    ori_mean = np.round(ori_mean, 2)

    with pd.ExcelWriter(save_path) as writer:
        for ex_i in range(result.shape[0]):
            sname = boost_method[ex_i]
            res = result[ex_i]
            res_mean = np.mean(res, axis=0)
            res_mean = np.round(res_mean, 2)

            mean_percent = (res_mean - ori_mean) / ori_mean * 100
            mean_percent = mean_percent.astype(np.int)
            percent = (res - origin_acc) / origin_acc * 100
            percent = percent.astype(np.int)
            res = np.concatenate([res, res_mean.reshape(1, -1)], axis=0)
            percent = np.concatenate([percent, mean_percent.reshape(1, -1)], axis=0)

            res_df = pd.DataFrame(res, columns=column, index=datasets)
            print(res.shape)
            for i in range(res_df.shape[0]):
                for j in range(res_df.shape[1]):
                    if percent[i, j] > 0:
                        prefix = '\\uparrow'
                    elif percent[i, j] < 0:
                        prefix = '\\downarrow'
                    elif percent[i,j] == 0:
                        prefix = ''
                    percent[i, j] = abs(percent[i, j])
                    res_df.iloc[i, j] = f'{res_df.iloc[i, j]}$^' + '{' + prefix + str(percent[i, j]) + '\\%}$'
                    if flag_arr[ex_i, i, j] == 1:
                        res_df.iloc[i, j] = f"\\textbf{{{res_df.iloc[i, j]}}}"
                    # elif flag_arr[ex_i, i, j] == -1:
                    #     res_df.iloc[i, j] = f"{res_df.iloc[i, j]}$^" + '{' + '\\boldmath{\\downarrow}}$'
            res_df.iloc[:, :] = res_df.iloc[:, :].applymap(lambda x: '&{}'.format(x))
            res_df.iloc[:, -1] = res_df.iloc[:, -1].apply(lambda x: '{}\\\\'.format(x))
            res_df.to_excel(writer, sheet_name=sname)

    # calculate Max Enhancement
    # save_path = '../../table/MaxEnhancePercent.xlsx'
    # with pd.ExcelWriter(save_path) as writer:
    #     final_res = []
    #     average_res = []
    #     for ex_i in range(result.shape[0]):
    #         sname = 'MaxEnhancePercent'
    #         res = result[ex_i]
    #         print(res.shape)
    #         res = np.round(res, 3)
    #         res = np.mean(res, axis=0)
    #         average_res.append(res)
    #         res = (res - ori_mean) / ori_mean * 100
    #         res = np.round(res, 1)
    #         final_res.append(res)
    #     final_res = np.array(final_res)
    #     average_res = np.array(average_res)
    #     average_res = np.round(average_res, 3)
    #     final_res = np.concatenate([final_res, average_res], axis=0)
    #     res_df = pd.DataFrame(final_res, columns=column, index=method * 2)
    #     for i in [0, 3]:
    #         for j in range(10):
    #             res_df.iloc[i, j] = f"\\textbf{{{res_df.iloc[i, j]}}}"
    #     res_df.iloc[:, :] = res_df.iloc[:, :].applymap(lambda x: '&{}'.format(x))
    #     res_df.iloc[:3, :] = res_df.iloc[:3, :].applymap(lambda x: '{}\%'.format(x))
    #     res_df.iloc[:, -1] = res_df.iloc[:, -1].apply(lambda x: '{}\\\\'.format(x))
    #     res_df.to_excel(writer, sheet_name=sname)


def plot_absolution_density(res, datasets, path, metrics, clustering_method):
    boost_method = ['HIOH', 'HIOH-NoAvg']
    plotAbsoluteDensityMeanGraph(res, boost_method, path, metrics, clustering_method, 'mean',
                          'absolution-density')

if __name__ == "__main__":
    global map_char, count
    count = 1
    map_char = {}
    generateMap(map_char)
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../../result-final/', type=str, help='the director to save results')
    parser.add_argument('--save_fig', default='../../figure/absolution-density/', type=str, help='the director to save results')
    parser.add_argument('--dir', default='../../result-3/', type=str, help='the director of dataset')
    parser.add_argument('--data_name', default='htru', type=str,
                        help='dataset name, one of {}')
    parser.add_argument('--itr', default=5, type=float, help='iteration times')
    parser.add_argument('--r_time', default=5, type=int, help='five times the average distance between all objects '
                                                              'in X and their closest neighbors')
    parser.add_argument('--ratio', default=1, type=float)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--show_fig', default=True, type=bool)
    parser.add_argument('--NoEnhance', default=False, type=bool)

    args = parser.parse_args()
    # dm_start = 5
    # dm_params = 15
    # segment = 2
    idx_list = [i for i in range(0, 46)]
    ratios = [1]
    clustering_method = ['kmeans', 'agg']
    # clustering_method = ['dbscan']
    # boost_method = ['DM-averaging-fuzz-center-NLOGN-SKD', 'SBCA', 'HIBOG', 'HIAC', 'Herd'] # 'DM-NoAvg-KNN'
    boost_method = ['DM-averaging-fuzz-center-NLOGN-SKD', 'DM-NoAvg-formal-SKD']
    # boost_method = [ 'DM-averaging-fuzz-farthest-formal'
    #                 , 'DM-averaging-fuzz-lowest-formal', 'DM-averaging-fuzz-highest-formal',
    #                 'DM-averaging-fuzz-random-formal', 'DM-averaging-fuzz-nearest-formal', 'DM-averaging-fuzz-center-NLOGN-SKD'
    #                 ] # 'DM-NoAvg-KNN'' DM-averaging-fuzz-center-new',
    # boost_method = ['DM-averaging-fuzz-highest-formal', 'DM-averaging-fuzz-lowest-formal',
    #                 'DM-averaging-fuzz-center-NLOGN-SKD']
    # data_name_set = ['Balance', 'Tae', 'Seeds', 'Yeast', 'Redwine', 'Wireless', 'Waveform', 'Whitewine', 'Abalone']#
    # data_name_set = ['Redwine', 'Waveform', 'Wireless', 'Tae', 'Seeds', 'Yeast']
    data_name_set = ['Letter', 'Archive', 'HAPT', 'Redwine', 'Bean'
                    , 'Wireless', 'Balance', 'Htru', 'Tae', 'Yeast']#
    # data_name_set = ['tae', 'redwine', 'balance', 'yeast', 'wireless', 'HAPT', 'bean', 'Letter', 'archive']
    metrics = ['ARI', 'NMI', 'FMI', 'PUR', 'VM']

    # method = ['HIOH-F', 'HIOH-L', 'HIOH-H', 'HIOH-R', 'HIOH-N', 'HIOH']
    method = ['HIOH', 'HIOH-NoAvg']
    # method = ['HOP', 'Herd', 'SBCA', 'HIBOG', 'HIAC', 'Herd']
    plot_data_name = ['Let', 'Arc', 'Hap', 'Red', 'Bea', 'Wir', 'Bal', 'Htr',
                        'Tae', 'Yea']
    # plot_data_name = ['Tae', 'Red', 'Bal', 'Yea', 'Wir', 'Hap', 'Bea', 'Let', 'Arc']

    final_res_mean = []
    final_res_max = []

    for name in data_name_set:
        args.data_name = name
        cls_mean_name_list = []
        cls_max_name_list = []
        print("dataset: ", name)
        for cls in clustering_method:
            cls_mean_res_list = []
            cls_max_res_list = []
            for boost in boost_method:
                mth_mean_res_list = []
                mth_max_res_list = []
                for rate in ratios:
                    args.ratio = rate
                    print("boost method: ", boost)
                    if boost in ['DM-averaging-fuzz-center-NLOGN-SKD', 'DM-averaging-fuzz', 'DM-NoAvg-formal-SKD',
                                 'DM-averaging-fuzz-farthest-formal', 'DM-averaging-fuzz-nearest-formal'
                                 , 'DM-averaging-fuzz-lowest-formal', 'DM-averaging-fuzz-highest-formal',
                                 'DM-averaging-fuzz-random-formal', 'DM-averaging-fuzz-center-NLOGN',
                                 'DM-averaging-fuzz-center-new', 'DM-averaging-fuzz-center-NLOGN-SKD']:
                        file_path = args.save_dir + args.data_name + "/result-" + str(args.ratio) + "-" + boost + "-" + cls + ".csv"
                        res = pd.read_csv(file_path).values[11:61, 0:len(metrics)]
                        # res = res[idx_list]
                    elif boost in ['HIBOG', 'HIAC', 'SBCA', 'Herd']:
                        file_path = args.save_dir + args.data_name + "/result-" + str(args.ratio) + "-" + boost + "-" + cls + ".csv"
                        res = pd.read_csv(file_path).values[:, 0:len(metrics)]
                    if cls in ['kmeans', 'agg']:
                        mean_res = np.mean(res, axis=0)
                        max_res = np.max(res, axis=0)
                    elif cls in ['dbscan']:
                        length = res.shape[0]
                        cal_res = []
                        for i in range(length // 25):
                            tmp = res[i*25:(i+1)*25]
                            single_max = np.max(tmp, axis=0)
                            cal_res.append(single_max)
                        cal_res = np.array(cal_res)
                        max_res = np.max(cal_res, axis=0)
                        mean_res = np.mean(cal_res, axis=0)
                    elif cls in ['birch', 'spectral']:
                        length = res.shape[0]
                        cal_res = []
                        for i in range(length // 10):
                            tmp = res[i*10:(i+1)*10]
                            single_max = np.max(tmp, axis=0)
                            cal_res.append(single_max)
                        cal_res = np.array(cal_res)
                        max_res = np.max(cal_res, axis=0)
                        mean_res = np.mean(cal_res, axis=0)
                    mth_max_res_list.append(max_res)
                    mth_mean_res_list.append(mean_res)
                cls_max_res_list.append(mth_max_res_list)
                cls_mean_res_list.append(mth_mean_res_list)

            # cls_max_res = np.array(cls_max_res_list)
            # cls_mean_res = np.array(cls_mean_res_list)
            # print(cls_mean_res.shape)
            # # print(cls_max_res)
            # print(cls_mean_res)
            # if os.path.isdir(args.save_fig) is False:
            #     os.mkdir(args.save_fig)
            # path = args.save_fig + args.data_name + "/"
            # plotMetrics(cls_mean_res, name, boost_method, path, metrics, cls, 'mean')

            cls_mean_name_list.append(cls_mean_res_list)
            cls_max_name_list.append(cls_max_res_list)

        final_res_max.append(cls_max_name_list)
        final_res_mean.append(cls_mean_name_list)

    final_res_mean = np.array(final_res_mean)
    final_res_max = np.array(final_res_max)
    print(final_res_mean.shape)
    print(final_res_max.shape)
    res_mean = np.squeeze(final_res_mean, axis=-2)
    res_max = np.squeeze(final_res_max, axis=-2)
    print(res_mean.shape)

    # calculate_meanvalue_for_absolution_hopping(res_mean, metrics, clustering_method, args, '../../table/absolution-hopping.xlsx')
    # calculate_comparison_table(res_mean, boost_method, '../../table/comparison-baseline.xlsx')
    # calculate_comparison_table_maxima(res_max, boost_method)

    if os.path.isdir(args.save_fig) is False:
        os.mkdir(args.save_fig)
    path = args.save_fig
    plot_absolution_density(res_mean, data_name_set, path, metrics, clustering_method)

    # plotMetricsSingleRate(res_mean, method, path, metrics, clustering_method, 'mean')
    # plotMetricsSingleRate(res_max, method, path, metrics, clustering_method, 'max')
    # plotMetrics(cls_max_res, name, boost_method, path, metrics, cls, 'max')


