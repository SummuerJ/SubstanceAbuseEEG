import warnings
warnings.filterwarnings("ignore")
from pyts.image import MarkovTransitionField
import pandas as pd
import scipy.io as scio
import numpy as np
import h5py


def hjorth(xV):
    n = len(xV)
    mm = [0]
    dxV = np.diff(mm + xV)
    ddxV = np.diff(mm + dxV)
    mx2 = np.var(xV, ddof=1)
    mdx2 = np.var(dxV, ddof=1)
    mddx2 = np.var(ddxV, ddof=1)

    mob = mdx2 / mx2

    activity = mx2
    mobility = np.sqrt(mob)
    complexity = np.sqrt(mddx2 / mdx2) / mobility
    return activity, mobility, complexity

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

def save2h5(data_list,label_list):
    fre_data = np.array(data_list)
    label_np = np.array(label_list)
    print('数据集中原始的标签顺序是:\n', label_np)
    print('数据集中原始数据长度:\n', len(label_np))

    f = h5py.File('Dataset_PSD_train.h5', 'w')
    f['fre'] = fre_data
    f['labels'] = label_np
    f.close()

def save2np(sali, coca):
    salii = np.array(sali)
    cocaa = np.array(coca)
    print(salii.shape)
    print(cocaa.shape)
    np.save("ts_sali.npy", salii)
    np.save("ts_coca.npy", cocaa)
    # b = np.load("filename.npy")

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


sali_im_total = []
sali_im_count = 0
coca_im_total = []
coca_im_count = 0

#  coca
coca = scio.loadmat('data/coca.mat')  # 5400*7
coca_data = coca['coca']
# arr = pd.DataFrame(dd)
data = []
for i in range(7):  # 7只小鼠
    alll = coca_data[i]
    X = np.array(alll[:540000])
    X = X.reshape(108000, 5)  # 降采样,1000Hz-->200Hz
    # X = pd.DataFrame(X)

    for j in range(0, 5):  # 降采样后的每一条都可以用作数据
        k = 0  # 用作划窗
        g = 0  # 用作命名
        # XX = list(X[j])
        XX = X[:, j]

        while k + 6000 <= 108000:  # 108000:    # 采取30s判断
            print(j)
            data = XX[k:(k + 6000)]
            k += 3000  # 采取15s划窗判断
            data = data.reshape(120, 50)  # PAA
            activity_list = []
            mobility_list = []
            complexity_list = []
            for linedata in data:
                activity, mobility, complexity = hjorth(linedata)
                activity_list.append(activity)
                mobility_list.append(mobility)
                complexity_list.append(complexity)

            activity_list = new_max_min(activity_list, -1, 1)
            mobility_list = new_max_min(mobility_list, -1, 1)
            complexity_list = new_max_min(complexity_list, -1, 1)

            activity_list = np.array(activity_list)
            mobility_list = np.array(mobility_list)
            complexity_list = np.array(complexity_list)

            activity_list1 = activity_list.reshape(1, -1)
            mobility_list1 = mobility_list.reshape(1, -1)
            complexity_list1 = complexity_list.reshape(1, -1)


            mtf1 = MarkovTransitionField(image_size=120)
            mtf2 = MarkovTransitionField(image_size=120)
            mtf3 = MarkovTransitionField(image_size=120)
            X_mtf1 = mtf1.fit_transform(activity_list1)
            X_mtf2 = mtf2.fit_transform(mobility_list1)
            X_mtf3 = mtf3.fit_transform(complexity_list1)


            X_im1 = ((X_mtf1[0] + 1) / 2) * 255
            X_im2 = ((X_mtf2[0] + 1) / 2) * 255
            X_im3 = ((X_mtf3[0] + 1) / 2) * 255

            immage = []
            for row in range(120):
                nnn = []
                for col in range(120):
                    swap = [X_im1[row][col], X_im2[row][col], X_im3[row][col]]
                    nnn.append(swap)
                immage.append(nnn)

            # print(immage)
            # print(X_im.shape)
            coca_im_total.append(immage)


# sali
sali = scio.loadmat('data/sali.mat')  # 5400*7
sali_data = sali['sali']
# arr = pd.DataFrame(dd)
data = []
for i in range(7):  # 7只小鼠
    alll = sali_data[i]
    X = np.array(alll[:540000])
    X = X.reshape(108000, 5)  # 降采样,1000Hz-->200Hz
    # X = pd.DataFrame(X)

    for j in range(0, 5):  # 降采样后的每一条都可以用作数据
        k = 0  # 用作划窗
        g = 0  # 用作命名
        # XX = list(X[j])
        XX = X[:, j]

        while k + 6000 <= 108000:  # 108000:    # 采取30s判断
            print(j)
            data = XX[k:(k + 6000)]
            k += 3000  # 采取15s划窗判断
            data = data.reshape(120, 50)  # PAA
            activity_list = []
            mobility_list = []
            complexity_list = []
            for linedata in data:
                activity, mobility, complexity = hjorth(linedata)
                activity_list.append(activity)
                mobility_list.append(mobility)
                complexity_list.append(complexity)

            activity_list = new_max_min(activity_list, -1, 1)
            mobility_list = new_max_min(mobility_list, -1, 1)
            complexity_list = new_max_min(complexity_list, -1, 1)

            activity_list = np.array(activity_list)
            mobility_list = np.array(mobility_list)
            complexity_list = np.array(complexity_list)

            activity_list1 = activity_list.reshape(1, -1)
            mobility_list1 = mobility_list.reshape(1, -1)
            complexity_list1 = complexity_list.reshape(1, -1)

            mtf1 = MarkovTransitionField(image_size=120)
            mtf2 = MarkovTransitionField(image_size=120)
            mtf3 = MarkovTransitionField(image_size=120)
            X_mtf1 = mtf1.fit_transform(activity_list1)
            X_mtf2 = mtf2.fit_transform(mobility_list1)
            X_mtf3 = mtf3.fit_transform(complexity_list1)

            X_im1 = ((X_mtf1[0] + 1) / 2) * 255
            X_im2 = ((X_mtf2[0] + 1) / 2) * 255
            X_im3 = ((X_mtf3[0] + 1) / 2) * 255


            immage = []
            for row in range(120):
                nnn = []
                for col in range(120):
                    swap = [X_im1[row][col], X_im2[row][col], X_im3[row][col]]
                    nnn.append(swap)
                immage.append(nnn)

            # print(immage)
            # print(X_im.shape)
            sali_im_total.append(immage)
print(len(sali_im_total))
print(len(coca_im_total))


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
        # X = pd.DataFrame(X)

        for jj in range(0, 5):  # 降采样后的每一条都可以用作数据
            k = 0  # 用作划窗
            g = 0  # 用作命名
            XX = X[:, jj]

            while k + 6000 <= 120000:  # 108000:    # 采取30s判断
                print(jj)
                data = XX[k:(k + 6000)]
                k += 3000  # 采取15s划窗判断
                data = data.reshape(120, 50)  # PAA
                activity_list = []
                mobility_list = []
                complexity_list = []
                for linedata in data:
                    activity, mobility, complexity = hjorth(linedata)
                    activity_list.append(activity)
                    mobility_list.append(mobility)
                    complexity_list.append(complexity)

                activity_list = new_max_min(activity_list, -1, 1)
                mobility_list = new_max_min(mobility_list, -1, 1)
                complexity_list = new_max_min(complexity_list, -1, 1)

                activity_list = np.array(activity_list)
                mobility_list = np.array(mobility_list)
                complexity_list = np.array(complexity_list)

                activity_list1 = activity_list.reshape(1, -1)
                mobility_list1 = mobility_list.reshape(1, -1)
                complexity_list1 = complexity_list.reshape(1, -1)

                mtf1 = MarkovTransitionField(image_size=120)
                mtf2 = MarkovTransitionField(image_size=120)
                mtf3 = MarkovTransitionField(image_size=120)
                X_mtf1 = mtf1.fit_transform(activity_list1)
                X_mtf2 = mtf2.fit_transform(mobility_list1)
                X_mtf3 = mtf3.fit_transform(complexity_list1)

                X_im1 = ((X_mtf1[0] + 1) / 2) * 255
                X_im2 = ((X_mtf2[0] + 1) / 2) * 255
                X_im3 = ((X_mtf3[0] + 1) / 2) * 255

                immage = []
                for row in range(120):
                    nnn = []
                    for col in range(120):
                        swap = [X_im1[row][col], X_im2[row][col], X_im3[row][col]]
                        nnn.append(swap)
                    immage.append(nnn)

                # print(immage)
                # print(X_im.shape)
                sali_im_total.append(immage)
# coca
for day in range(5):
    for mouse in range(4):
        pend_data = data2[coca_day[day]][mouse]
        X = np.array(pend_data[:600000])
        X = X.reshape(120000, 5)  # 降采样,1000Hz-->200Hz
        # X = pd.DataFrame(X)

        for jj in range(0, 5):  # 降采样后的每一条都可以用作数据
            k = 0  # 用作划窗
            g = 0  # 用作命名
            XX = X[:, jj]

            while k + 6000 <= 120000:  # 108000:    # 采取30s判断
                print(jj)
                data = XX[k:(k + 6000)]
                k += 3000  # 采取15s划窗判断
                data = data.reshape(120, 50)  # PAA
                activity_list = []
                mobility_list = []
                complexity_list = []
                for linedata in data:
                    activity, mobility, complexity = hjorth(linedata)
                    activity_list.append(activity)
                    mobility_list.append(mobility)
                    complexity_list.append(complexity)

                activity_list = new_max_min(activity_list, -1, 1)
                mobility_list = new_max_min(mobility_list, -1, 1)
                complexity_list = new_max_min(complexity_list, -1, 1)

                activity_list = np.array(activity_list)
                mobility_list = np.array(mobility_list)
                complexity_list = np.array(complexity_list)

                activity_list1 = activity_list.reshape(1, -1)
                mobility_list1 = mobility_list.reshape(1, -1)
                complexity_list1 = complexity_list.reshape(1, -1)

                mtf1 = MarkovTransitionField(image_size=120)
                mtf2 = MarkovTransitionField(image_size=120)
                mtf3 = MarkovTransitionField(image_size=120)
                X_mtf1 = mtf1.fit_transform(activity_list1)
                X_mtf2 = mtf2.fit_transform(mobility_list1)
                X_mtf3 = mtf3.fit_transform(complexity_list1)

                X_im1 = ((X_mtf1[0] + 1) / 2) * 255
                X_im2 = ((X_mtf2[0] + 1) / 2) * 255
                X_im3 = ((X_mtf3[0] + 1) / 2) * 255

                immage = []
                for row in range(120):
                    nnn = []
                    for col in range(120):
                        swap = [X_im1[row][col], X_im2[row][col], X_im3[row][col]]
                        nnn.append(swap)
                    immage.append(nnn)

                # print(immage)
                # print(X_im.shape)
                coca_im_total.append(immage)

print(len(sali_im_total))
print(len(coca_im_total))


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
        # X = pd.DataFrame(X)

        for jj in range(0, 5):  # 降采样后的每一条都可以用作数据
            k = 0  # 用作划窗
            g = 0  # 用作命名
            XX = X[:, jj]

            while k + 6000 <= 120000:  # 108000:    # 采取30s判断
                print(jj)
                data = XX[k:(k + 6000)]
                k += 3000  # 采取15s划窗判断
                data = data.reshape(120, 50)  # PAA
                activity_list = []
                mobility_list = []
                complexity_list = []
                for linedata in data:
                    activity, mobility, complexity = hjorth(linedata)
                    activity_list.append(activity)
                    mobility_list.append(mobility)
                    complexity_list.append(complexity)

                activity_list = new_max_min(activity_list, -1, 1)
                mobility_list = new_max_min(mobility_list, -1, 1)
                complexity_list = new_max_min(complexity_list, -1, 1)

                activity_list = np.array(activity_list)
                mobility_list = np.array(mobility_list)
                complexity_list = np.array(complexity_list)

                activity_list1 = activity_list.reshape(1, -1)
                mobility_list1 = mobility_list.reshape(1, -1)
                complexity_list1 = complexity_list.reshape(1, -1)

                mtf1 = MarkovTransitionField(image_size=120)
                mtf2 = MarkovTransitionField(image_size=120)
                mtf3 = MarkovTransitionField(image_size=120)
                X_mtf1 = mtf1.fit_transform(activity_list1)
                X_mtf2 = mtf2.fit_transform(mobility_list1)
                X_mtf3 = mtf3.fit_transform(complexity_list1)

                X_im1 = ((X_mtf1[0] + 1) / 2) * 255
                X_im2 = ((X_mtf2[0] + 1) / 2) * 255
                X_im3 = ((X_mtf3[0] + 1) / 2) * 255

                immage = []
                for row in range(120):
                    nnn = []
                    for col in range(120):
                        swap = [X_im1[row][col], X_im2[row][col], X_im3[row][col]]
                        nnn.append(swap)
                    immage.append(nnn)

                # print(immage)
                # print(X_im.shape)
                sali_im_total.append(immage)
# coca
for day in range(5):
    for mouse in range(6):
        pend_data = data3[coca_day[day]][mouse]
        X = np.array(pend_data[:600000])
        X = X.reshape(120000, 5)  # 降采样,1000Hz-->200Hz
        # X = pd.DataFrame(X)

        for jj in range(0, 5):  # 降采样后的每一条都可以用作数据
            k = 0  # 用作划窗
            g = 0  # 用作命名
            XX = X[:, jj]

            while k + 6000 <= 120000:  # 108000:    # 采取30s判断
                print(jj)
                data = XX[k:(k + 6000)]
                k += 3000  # 采取15s划窗判断
                data = data.reshape(120, 50)  # PAA
                activity_list = []
                mobility_list = []
                complexity_list = []
                for linedata in data:
                    activity, mobility, complexity = hjorth(linedata)
                    activity_list.append(activity)
                    mobility_list.append(mobility)
                    complexity_list.append(complexity)

                activity_list = new_max_min(activity_list, -1, 1)
                mobility_list = new_max_min(mobility_list, -1, 1)
                complexity_list = new_max_min(complexity_list, -1, 1)

                activity_list = np.array(activity_list)
                mobility_list = np.array(mobility_list)
                complexity_list = np.array(complexity_list)

                activity_list1 = activity_list.reshape(1, -1)
                mobility_list1 = mobility_list.reshape(1, -1)
                complexity_list1 = complexity_list.reshape(1, -1)

                mtf1 = MarkovTransitionField(image_size=120)
                mtf2 = MarkovTransitionField(image_size=120)
                mtf3 = MarkovTransitionField(image_size=120)
                X_mtf1 = mtf1.fit_transform(activity_list1)
                X_mtf2 = mtf2.fit_transform(mobility_list1)
                X_mtf3 = mtf3.fit_transform(complexity_list1)

                X_im1 = ((X_mtf1[0] + 1) / 2) * 255
                X_im2 = ((X_mtf2[0] + 1) / 2) * 255
                X_im3 = ((X_mtf3[0] + 1) / 2) * 255

                immage = []
                for row in range(120):
                    nnn = []
                    for col in range(120):
                        swap = [X_im1[row][col], X_im2[row][col], X_im3[row][col]]
                        nnn.append(swap)
                    immage.append(nnn)

                # print(immage)
                # print(X_im.shape)
                coca_im_total.append(immage)

print(len(sali_im_total))
print(len(coca_im_total))

#sali
data4 = scio.loadmat('data/alldata/sali_cutted_6000.mat')  # cutted_data
# # print(len(data['cutted_data']))  # 40601
# # print(len(data['cutted_data'][0]))  # 6000
for idd in range(10000):
    pend_data = data4['cutted_data'][idd]
    # fre = psd(pend_data)
    # sali_frequency.append(fre)
    # data = XX[k:(k + 6000)]
    pend_data = pend_data.reshape(120, 50)  # PAA
    data = pd.DataFrame(pend_data)
    activity_list = []
    mobility_list = []
    complexity_list = []
    for linedata in data:
        activity, mobility, complexity = hjorth(linedata)
        activity_list.append(activity)
        mobility_list.append(mobility)
        complexity_list.append(complexity)

    activity_list = new_max_min(activity_list, -1, 1)
    mobility_list = new_max_min(mobility_list, -1, 1)
    complexity_list = new_max_min(complexity_list, -1, 1)

    activity_list = np.array(activity_list)
    mobility_list = np.array(mobility_list)
    complexity_list = np.array(complexity_list)

    activity_list1 = activity_list.reshape(1, -1)
    mobility_list1 = mobility_list.reshape(1, -1)
    complexity_list1 = complexity_list.reshape(1, -1)

    mtf1 = MarkovTransitionField(image_size=120)
    mtf2 = MarkovTransitionField(image_size=120)
    mtf3 = MarkovTransitionField(image_size=120)
    X_mtf1 = mtf1.fit_transform(activity_list1)
    X_mtf2 = mtf2.fit_transform(mobility_list1)
    X_mtf3 = mtf3.fit_transform(complexity_list1)

    X_im1 = ((X_mtf1[0] + 1) / 2) * 255
    X_im2 = ((X_mtf2[0] + 1) / 2) * 255
    X_im3 = ((X_mtf3[0] + 1) / 2) * 255

    immage = []
    for row in range(120):
        nnn = []
        for col in range(120):
            swap = [X_im1[row][col], X_im2[row][col], X_im3[row][col]]
            nnn.append(swap)
        immage.append(nnn)

    # print(immage)
    # print(X_im.shape)
    sali_im_total.append(immage)

print(len(sali_im_total))
print(len(coca_im_total))

sali_im_count = len(sali_im_total)
coca_im_count = len(coca_im_total)

# train_data = sali_frequency + coca_frequency
# train_data_label = [0]*len(sali_frequency) +[1]*len(coca_frequency)
# save2h5(train_data, train_data_label)

# save2np(sali_im_total, coca_im_total)  # (18050, 50) (10975, 50)
save_image_to_h5py('20210410_Dataset_hjorth_MTF_120x120x3.h5')