from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import scipy.io as scio
import pandas as pd
import cv2
import h5py



def save2h5(data_list, label_list):
    fre_data = np.array(data_list)
    label_np = np.array(label_list)
    print('数据集中原始的标签顺序是:\n', label_np)
    print('数据集中原始数据长度:\n', len(label_np))

    f = h5py.File('20210519_Dataset_200x10.h5', 'w')
    f['fre'] = fre_data
    f['labels'] = label_np
    f.close()

def save2np(sali, coca):
    salii = np.array(sali)
    cocaa = np.array(coca)
    print(salii.shape)
    print(cocaa.shape)
    np.save("psd_sali.npy", salii)
    np.save("psd_coca.npy", cocaa)
    # b = np.load("filename.npy")


sali_6000_data = []
coca_6000_data = []

#  coca
coca = scio.loadmat('data/coca.mat')  # 5400*7
coca_data = coca['coca']
# arr = pd.DataFrame(dd)
data = []
for i in range(7):  # 7只小鼠
    alll = coca_data[i]
    X = np.array(alll[:540000])
    X = X.reshape(108000, 5)  # 降采样,1000Hz-->200Hz
    X = pd.DataFrame(X)

    for j in range(0, 5):  # 降采样后的每一条都可以用作数据
        k = 0  # 用作划窗
        g = 0  # 用作命名
        XX = list(X[j])

        while k + 2000 <= 108000:  # 108000:    # 采取30s判断
            print(j)
            data = XX[k:(k + 2000)]
            data = np.array(data)
            data = data.reshape(200, 10)
            k += 2000  # 采取15s划窗判断
            coca_6000_data.append(data)

print(len(sali_6000_data))
print(len(coca_6000_data))

# sali
sali = scio.loadmat('data/sali.mat')  # 5400*7
sali_data = sali['sali']
# arr = pd.DataFrame(dd)
data = []
for i in range(7):  # 7只小鼠
    alll = sali_data[i]
    X = np.array(alll[:540000])
    X = X.reshape(108000, 5)  # 降采样,1000Hz-->200Hz
    X = pd.DataFrame(X)

    for j in range(0, 5):  # 降采样后的每一条都可以用作数据
        k = 0  # 用作划窗
        g = 0  # 用作命名
        XX = list(X[j])

        while k + 2000 <= 108000:  # 108000:    # 采取30s判断
            print(j)
            data = XX[k:(k + 2000)]
            data = np.array(data)
            data = data.reshape(200, 10)
            k += 2000  # 采取15s划窗判断
            sali_6000_data.append(data)
print(len(sali_6000_data))
print(len(coca_6000_data))

# alldata
data2 = scio.loadmat('data/alldata/alldata.mat')  # coca_Day1-5;coca_pre_Day1-5;sali_Day1-5;sali_pre_Day1-5   data['coca_Day1'][0-3]取第一天的第一只
coca_day = ["coca_Day1", "coca_Day2", "coca_Day3", "coca_Day4", "coca_Day5"]
sali_day = ["sali_Day1", "sali_Day2", "sali_Day3", "sali_Day4", "sali_Day5"]

# sali
for day in range(0, 5):
    for mouse in range(0, 3):
        # print(type(data))
        pend_data = data2[sali_day[day]][mouse]
        X = np.array(pend_data[:600000])
        X = X.reshape(120000, 5)  # 降采样,1000Hz-->200Hz
        X = pd.DataFrame(X)

        for jj in range(0, 5):  # 降采样后的每一条都可以用作数据
            k = 0  # 用作划窗
            g = 0  # 用作命名
            XX = list(X[jj])

            while k + 2000 <= 120000:  # 108000:    # 采取30s判断
                print(jj)
                data = XX[k:(k + 2000)]
                data = np.array(data)
                data = data.reshape(200, 10)
                k += 2000  # 采取15s划窗判断
                sali_6000_data.append(data)
# coca
for day in range(5):
    for mouse in range(4):
        pend_data = data2[coca_day[day]][mouse]
        X = np.array(pend_data[:600000])
        X = X.reshape(120000, 5)  # 降采样,1000Hz-->200Hz
        X = pd.DataFrame(X)

        for jj in range(0, 5):  # 降采样后的每一条都可以用作数据
            k = 0  # 用作划窗
            g = 0  # 用作命名
            XX = list(X[jj])

            while k + 2000 <= 120000:  # 108000:    # 采取30s判断
                print(jj)
                data = XX[k:(k + 2000)]
                data = np.array(data)
                data = data.reshape(200, 10)
                k += 2000  # 采取15s划窗判断
                coca_6000_data.append(data)
print(len(sali_6000_data))
print(len(coca_6000_data))


# alldata
data3 = scio.loadmat('data/alldata/raw_all_data.mat')  # coca_Day1-5;coca_pre_Day1-5;sali_Day1-5;sali_pre_Day1-5   data['coca_Day1'][0-3]取第一天的第一只
coca_day = ["coca_Day1", "coca_Day2", "coca_Day3", "coca_Day4", "coca_Day5"]
sali_day = ["sali_Day1", "sali_Day2", "sali_Day3", "sali_Day4", "sali_Day5"]

# sali
for day in range(5):
    for mouse in range(4):
        pend_data = data3[sali_day[day]][mouse]
        X = np.array(pend_data[:600000])
        X = X.reshape(120000, 5)  # 降采样,1000Hz-->200Hz
        X = pd.DataFrame(X)

        for jj in range(0, 5):  # 降采样后的每一条都可以用作数据
            k = 0  # 用作划窗
            g = 0  # 用作命名
            XX = list(X[jj])

            while k + 2000 <= 120000:  # 108000:    # 采取30s判断
                print(jj)
                data = XX[k:(k + 2000)]
                data = np.array(data)
                data = data.reshape(200, 10)
                k += 2000  # 采取15s划窗判断
                sali_6000_data.append(data)
# coca
for day in range(5):
    for mouse in range(6):
        pend_data = data3[coca_day[day]][mouse]
        X = np.array(pend_data[:600000])
        X = X.reshape(120000, 5)  # 降采样,1000Hz-->200Hz
        X = pd.DataFrame(X)

        for jj in range(0, 5):  # 降采样后的每一条都可以用作数据
            k = 0  # 用作划窗
            g = 0  # 用作命名
            XX = list(X[jj])

            while k + 2000 <= 120000:  # 108000:    # 采取30s判断
                print(jj)
                data = XX[k:(k + 2000)]
                data = np.array(data)
                data = data.reshape(200, 10)
                k += 2000  # 采取15s划窗判断
                coca_6000_data.append(data)
print(len(sali_6000_data))
print(len(coca_6000_data))

#sali
data4 = scio.loadmat('data/alldata/sali_cutted_6000.mat')  # cutted_data
# # print(len(data['cutted_data']))  # 40601
# # print(len(data['cutted_data'][0]))  # 6000
for idd in range(4500):
    pend_data = data4['cutted_data'][idd]
    pend_data = pend_data[:2000]
    pend_data = np.array(pend_data)
    pend_data = pend_data.reshape(200, 10)
    sali_6000_data.append(pend_data)
print(len(sali_6000_data))
print(len(coca_6000_data))

# 3378
# 3378
# 16890
# 16890
train_data = sali_6000_data + coca_6000_data
train_data_label = [0]*len(sali_6000_data) +[1]*len(coca_6000_data)
train_data = np.array(train_data)
train_data_label = np.array(train_data_label)
# print(train_data)
print(train_data.shape)
save2h5(train_data, train_data_label)  # 4390

# save2np(sali_frequency, coca_frequency)  # (18050, 50) (10975, 50)
