import cv2
import h5py
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField

import numpy as np
import pandas as pd
import scipy.io as scio


def new_max_min(data_arr, a=-1, b=1):   # 归一化到[-1,1]

    # 最值
    min_value = min(data_arr)
    max_value = max(data_arr)

    # 权重
    addtioncoefficient = (a*max_value-b*min_value)/(max_value-min_value)
    Multicoefficient = (b-a) / (max_value - min_value)

    new = []

    for i in data_arr:
        new.append(i*Multicoefficient+addtioncoefficient)

    return new

# 加载数据集中的文件
def save_image_to_h5py(path):
    img_list = []
    label_list = []

    for dataa in coca_im_total:
        img_list.append(dataa)

    for dataa in sali_im_total:
        img_list.append(dataa)

    label_list = [1]*coca_im_count
    bsali = [0]*sali_im_count
    label_list.extend(bsali)

    img_np = np.array(img_list)
    label_np = np.array(label_list)
    print(img_np.shape)  # (29025, 50, 50, 3)
    print(label_np.shape)  # (29025,)
    print('数据集中原始的标签顺序是:\n', label_np)

    f = h5py.File(path, 'w')
    f['image'] = img_np
    f['labels'] = label_np
    f.close()

# '''
# 读取时间序列的数据
# 怎么读取需要你自己写
# X为ndarray类型数据
# '''
# # Transform the time series into Gramian Angular Fields
# gasf = GramianAngularField(image_size=28, method='summation')
# X_gasf = gasf.fit_transform(X)
# gadf = GramianAngularField(image_size=28, method='difference')
# X_gadf = gadf.fit_transform(X)

'''
读取时间序列的数据
怎么读取需要你自己写
'''
coca_data = np.load("psd_coca.npy")  # (10975, 50)
sali_data = np.load("psd_sali.npy")  # 18050,50


# GASF
# Show the results for the first time series
sali_im_total = []
sali_im_count = 0
coca_im_total = []
coca_im_count = 0


for sali_D in sali_data:
    data = new_max_min(sali_D, -1, 1)
    data = np.array(data)
    data = data.reshape(1, -1)
    mtf = MarkovTransitionField(image_size=50)
    X_mtf = mtf.fit_transform(data)
    X_im = ((X_mtf[0] + 1) / 2) * 255
    X_im = X_im.astype(np.uint8)
    X_im = cv2.cvtColor(X_im, cv2.COLOR_GRAY2BGR)
    # print(X_im.shape)
    sali_im_total.append(X_im)

    # # image show
    # plt.imshow(X_mtf[0], cmap='binary', origin='lower')
    # plt.title("mtf", fontsize=16)
    # plt.tight_layout()
    # plt.show()

for coca_D in coca_data:
    data2 = new_max_min(coca_D, -1, 1)
    data2 = np.array(data2)
    data2 = data2.reshape(1, -1)
    mtf = MarkovTransitionField(image_size=50)
    X_mtf = mtf.fit_transform(data2)
    X_im = ((X_mtf[0] + 1) / 2) * 255
    X_im = X_im.astype(np.uint8)
    X_im = cv2.cvtColor(X_im, cv2.COLOR_GRAY2BGR)
    # print(X_im.shape)
    coca_im_total.append(X_im)
    # image show
    # plt.imshow(X_gasf[0], cmap='binary', origin='lower')
    # plt.title("GASF", fontsize=16)
    # plt.tight_layout()
    # plt.show()


sali_im_count = len(sali_data)
coca_im_count = len(coca_data)

save_image_to_h5py('20210406_Dataset_MTF_50x50x3.h5')








