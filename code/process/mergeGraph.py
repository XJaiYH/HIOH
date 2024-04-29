import os
import inspect

import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import glob, os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

prefix = 'flow'
num = 24
img_np = []
name_ls = ['Unbalance', 'Unbalance2', 'Unbalance3', 'Unbalance4', 'G2', 'S1']
ms_ls = ['ori', 'HIBOG', 'HIAC', 'SBCA', 'DM']
# sz代表图片像素
sz = (1200, 1050)
for ms in ms_ls:
    for name in name_ls:
        path = '../../paper/graph/densityGraph/' + name + '-' + ms + '.png'
        im = cv2.imread(path)
        # im.show()
        im = cv2.resize(im, sz)
        # im.show()
        # im_array = np.atleast_2d(im)
        # print(type(im_array), im_array.shape)
        img_np.append(im)

# 画布大小figsize根据sz和rows cols共同确定，即画布横向：画布纵向 = sz的长*cols ：sz的宽*rows
fig = plt.figure(figsize=(30, 14.5))

rows = 4
cols = 6
name_ls = ['Unbalance1', 'Unbalance2', 'Unbalance3', 'Unbalance4', 'G2-U', 'S1-U']
ms_ls = ['Original', 'HIBOG', 'HIAC', 'HIOH']
for i in range(1, rows * cols + 1):
    img_array = img_np[i - 1]
    # 子图位置
    ax = fig.add_subplot(rows, cols, i)
    ax.set_frame_on(False)
    # plt.axis('off')  # 去掉每个子图的坐标轴
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    if i <= 6:
        ax.set_title(name_ls[i-1], size=30, family='Times New Roman')
    if i in [1, 7, 13, 19, 25]:
        ax.set_ylabel(ms_ls[int((i-1)//6)], size=30, family='Times New Roman')
    img_array = img_array[:, :, ::-1]
    plt.imshow(img_array)

plt.subplots_adjust(wspace=0, hspace=0)  # 修改子图之间的间隔
plt.savefig('densityHeatMap.png', dpi=300)