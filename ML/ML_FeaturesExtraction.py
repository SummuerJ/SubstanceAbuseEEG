#-------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#
# Name:         Machine learning & Features Extraction
# Description: Based on tsfresh(incoporate Statistic\CWT\FFTC\En and so forth) and PSD and Hjorth and Entropy and RP
# Author:       jiexia
# Date:         2021/4/20
#-------------------------------------------------------------------------------

'''
记录一下：
时域方面，可提取均值、方差、极值等特征、hjorth🐶
频域方面，可提取功率谱，功率密度比，中值频率，平均功率频率等特征
脑电信号（EEG）：峰值、熵值、非线性能量、QRS波的峰值、波长等、QT间隔、ST间隔等统计特征；
基于模型的AR
线性变换中：PCA 和小波变换
分形理论用于特征提取时, 主要是针对非线性信号, 是用它的定量分析指标分维数作来特征矢量
'''

# import os
# import numpy as np
# import pandas as pd
# import time
# import datetime
# from tsfresh import extract_relevant_features
# from tsfresh import extract_features
import tsfresh as tsf
from scipy.fftpack import fft, fftshift, ifft
# from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# import scipy.io as scio
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# import cv2
import h5py
from pyunicorn.timeseries import RecurrencePlot
# from sampen import sampen2
import scipy.stats
import itertools as it
from math import e
import math

# pandas 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 读取数据
# len6000data_DAE.py
# 加载hdpy成np的形式
def load_dataset(path):

    h5_file = h5py.File(path, 'r')
    eegdatas = h5_file['fre']
    labels = h5_file['labels']

    return eegdatas, labels

#############################################################
# 特征函数
#############################################################

# 返回中心频率,带宽,也可返回0-50hz的功率谱值
band_delta = np.array(list(range(1, 5)))
band_theta = np.array(list(range(4, 9)))
band_alpha = np.array(list(range(8, 14)))
band_beta = np.array(list(range(13, 36)))
band_gamma = np.array(list(range(35, 50)))
band_all = np.array(list(range(1, 50)))
def psd(ts):
    fs = 200
    # 采样点数
    num_fft = 256
    Y = fft(ts, num_fft)  # FFT(Fast Fourier Transformation)快速傅里叶变换
    Y = np.abs(Y)
    # plt.plot(20*np.log10(Y[:50]))  # num_fft//2
    ps = Y ** 2 / num_fft  # 直接平方、功率谱 power spectrum

    pxx = np.array(ps[:50])

    central_frequency_delta = np.sum(np.dot(band_delta, pxx[1:5]), axis=0) / np.sum(pxx[1:5], axis=0)
    central_frequency_theta = np.sum(np.dot(band_theta, pxx[4:9]), axis=0) / np.sum(pxx[4:9], axis=0)
    central_frequency_alpha = np.sum(np.dot(band_alpha, pxx[8:14]), axis=0) / np.sum(pxx[8:14], axis=0)
    central_frequency_beta = np.sum(np.dot(band_beta, pxx[13:36]), axis=0) / np.sum(pxx[13:36], axis=0)
    central_frequency_gamma = np.sum(np.dot(band_gamma, pxx[35:50]), axis=0) / np.sum(pxx[35:50], axis=0)
    central_frequency_all = np.sum(np.dot(band_all, pxx[1:50]), axis=0) / np.sum(pxx[1:50], axis=0)


    bandwidth_delta = np.sqrt(np.sum(np.dot(pow((band_delta - central_frequency_delta), 2), pxx[1:5]), axis=0) / np.sum(pxx[1:5], axis=0))
    bandwidth_theta = np.sqrt(np.sum(np.dot(pow((band_theta - central_frequency_theta), 2), pxx[4:9]), axis=0) / np.sum(pxx[4:9], axis=0))
    bandwidth_alpha = np.sqrt(np.sum(np.dot(pow((band_alpha - central_frequency_alpha), 2), pxx[8:14]), axis=0) / np.sum(pxx[8:14], axis=0))
    bandwidth_beta = np.sqrt(np.sum(np.dot(pow((band_beta - central_frequency_beta), 2), pxx[13:36]), axis=0) / np.sum(pxx[13:36], axis=0))
    bandwidth_gamma = np.sqrt(np.sum(np.dot(pow((band_gamma - central_frequency_gamma), 2), pxx[35:50]), axis=0) / np.sum(pxx[35:50], axis=0))
    bandwidth_all = np.sqrt(
        np.sum(np.dot(pow((band_all - central_frequency_all), 2), pxx[1:50]), axis=0) / np.sum(pxx[1:50], axis=0))

    return central_frequency_delta, central_frequency_theta, central_frequency_alpha, central_frequency_beta,\
        central_frequency_gamma, central_frequency_all, bandwidth_delta, bandwidth_theta, bandwidth_alpha, \
           bandwidth_beta, bandwidth_gamma, bandwidth_all



# 输出5个节律的功率谱密度
iter_freqs = [
        {'name': 'delta', 'fmin': 0.1, 'fmax': 3.75},
        {'name': 'theta', 'fmin': 3.75, 'fmax': 7.5},
        {'name': 'alpha', 'fmin': 7.5, 'fmax': 12.5},
        {'name': 'beta', 'fmin': 12.5, 'fmax': 35},
        {'name': 'gamma', 'fmin': 35, 'fmax': 49.9},
        {'name': 'beta_L', 'fmin': 12.5, 'fmax': 16},
        {'name': 'beta_M', 'fmin': 16.5, 'fmax': 20.5},
        {'name': 'beta_H', 'fmin': 20, 'fmax': 28},
        {'name': 'total', 'fmin': 0.1, 'fmax': 49.9}
]
def PSD(ts):
    psds, freqs = plt.psd(ts, len(ts), Fs=200, scale_by_freq=0)
    # print(psds)
    # print(len(psds))

    # 初始化能量列表
    eventEnergy = []
    # 遍历不同频率区间的能量和
    for iter_freq in iter_freqs:
        # eventEnergy.append(np.sum(psds[iter_freq['fmin']:(iter_freq['fmax']+1)]))
        eventEnergy.append(np.sum(psds[(iter_freq['fmin'] < freqs) & (freqs < iter_freq['fmax'])]))

    return eventEnergy

def PSDandDE(ts):
    psds, freqs = plt.psd(ts, len(ts), Fs=200, scale_by_freq=0)
    # print(psds)
    # print(len(psds))

    # 初始化能量列表
    eventEnergy = []
    # 遍历不同频率区间的能量和
    for iter_freq in iter_freqs:
        # eventEnergy.append(np.sum(psds[iter_freq['fmin']:(iter_freq['fmax']+1)]))
        eventEnergy.append(np.sum(psds[(iter_freq['fmin'] < freqs) & (freqs < iter_freq['fmax'])]))

    DE = []
    for data in eventEnergy:
        DE.append(np.log2(data))

    return eventEnergy[0], eventEnergy[1], eventEnergy[2], eventEnergy[3], eventEnergy[4], eventEnergy[5], \
           eventEnergy[6], eventEnergy[7], eventEnergy[8], DE[0], DE[1], DE[2], DE[3], DE[4], DE[5], DE[6], DE[7], DE[8]

def standardize(ts):
    """ Standardize data """

    # Standardize train and test
    std_ts = (ts - np.mean(ts, axis=0)[ :]) / np.std(ts, axis=0)[:]

    return std_ts

def statisticFeatures(ts):
    # 均值
    averagee = np.mean(ts)

    # 方差
    varr = np.var(ts)

    # 协方差
    covv = np.cov(ts)

    # Z-Score标准化
    ts = np.array(ts)
    ts = ts.reshape(-1, 1)
    stand_ts = standardize(ts)  # 标准化处理
    # 峰值
    maxx = np.max(stand_ts[0])

    return averagee, varr, covv, maxx


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


# 可以用近似熵来量化时间序列的规律性和波动的不可预测性。近似熵越高，意味着预测难度越大。
# 样本熵与近似熵类似，但在不同的复杂度上更具有一致性，即使是小型时间序列。例如，相比于“有规律”的时间序列，一个数据点较少的随机时间序列的近似熵较低，但一个较长的随机时间序列却有较高的近似熵。
def SampEn(U, m, r):
    """Compute Sample entropy"""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))


# print(SampEn(ss.value, m=2, r=0.2*np.std(ss.value)))      # 0.78

def Fuzzy_Entropy(x, m, r=0.25, n=2):
    """
    模糊熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    n 计算模糊隶属度时的维度
    """
    x = np.array(x)  # 将x转化为数组

    if x.ndim != 1:
        raise ValueError("x的维度不是一维")  # 检查x是否为一维数据

    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")  # 计算x的行数是否小于m+1

    entropy = 0  # 将x以m为窗口进行划分
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与其他行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        # 计算模糊隶属度
        D = np.exp(-np.power(D_value, n)/r)
        # 计算所有隶属度的平均值
        Lm = np.average(D.ravel())
        entropy = abs(entropy) - Lm

    return entropy

# 排列熵
def pec(y, D, t):
    y_len = len(y)
    serial = np.arange(0, D)
    y_perm = list(it.permutations(serial, D))
    DI = len(y_perm)
    count = np.zeros(DI)

    for i in range(y_len-(D-1)*t):
        y_x = np.argsort(y[i:i+t*D:t])

        for j in range(len(y_perm)):
            if tuple(y_x) == y_perm[j]:
                count[j] += 1

    pe = scipy.stats.entropy(count / (y_len-(D-1)*t), base=e)/math.log(DI)
    return pe

def S_entropy(counts):
    # Compute entropy.
    counts = np.array(counts)
    ps = counts/float(np.sum(counts))  # coerce to float and normalize
    ps = ps[np.nonzero(ps)]            # toss out zeros
    h = -np.sum(ps * np.log2(ps))   # compute entropy
    return h

"""
# 一阶差分绝对和、各阶自相关系数的聚合统计特征、 
ADF 检测统计值、峰度、时序数据复杂度、线性回归分析、
分组熵、近似熵、傅里叶变换频谱统计量、傅里叶变换系数、小坡变换
"""
def tsfresh_get_features(ts):
    # 峰度
    peak = tsf.feature_extraction.feature_calculators.kurtosis(ts)
    peak = round(peak, 2)

    # 一阶差分绝对和
    abs_sum = tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(ts)
    abs_sum = round(abs_sum, 2)

    # 各阶自相关系数的聚合统计特征
    param_statis = [{'f_agg': 'mean', 'maxlag': 2}]
    diff_statis = tsf.feature_extraction.feature_calculators.agg_autocorrelation(ts, param_statis)
    diff_statis = diff_statis[0][1]
    diff_statis = round(diff_statis, 2)

    # 自回归性
    auto_co = tsf.feature_extraction.feature_calculators.autocorrelation(ts, 33)
    auto_co = round(auto_co, 2)

    # 时序数据非线性度量
    c3 = tsf.feature_extraction.feature_calculators.c3(ts, 33)
    c3 = round(c3, 2)

    # ADF 检测统计值
    param_adf = [{'attr': 'pvalue'}]
    adf = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, param_adf)
    adf = adf[0][1]
    adf = round(adf, 2)

    # 时序数据复杂度
    complexity = tsf.feature_extraction.feature_calculators.cid_ce(ts, True)
    complexity = round(complexity, 2)

    # 线性回归分析
    param_line = [{'attr': 'pvalue'}]
    line = tsf.feature_extraction.feature_calculators.linear_trend(ts, param_line)
    line = list(zip(line))[0][0][1]
    line = round(line, 2)

    # 分组熵
    bin_entropy = tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)
    bin_entropy = round(bin_entropy, 2)

    # 近似熵
    appro_entropy = tsf.feature_extraction.feature_calculators.approximate_entropy(ts, 2, 0.2)
    appro_entropy = round(appro_entropy, 2)

    # 傅里叶变换频谱统计量
    param_fly = [{'aggtype': 'skew'}]
    fly = tsf.feature_extraction.feature_calculators.fft_aggregated(ts, param_fly)
    fly = list(zip(fly))[0][0][1]
    fly = round(fly, 2)

    # 傅里叶变换系数
    param_fly_change = [{'coeff': 2, 'attr': 'angle'}]
    fly_change = tsf.feature_extraction.feature_calculators.fft_coefficient(ts, param_fly_change)
    fly_change = list(zip(fly_change))[0][0][1]
    fly_change = round(fly_change, 2)  # [“centroid”, “variance”, “skew”, “kurtosis”]

    # 小坡变换 Ricker小波分析
    param_cwt = [{'widths': tuple([2, 2, 2]), 'coeff': 2, 'w': 2}]
    cwt = tsf.feature_extraction.feature_calculators.cwt_coefficients(ts, param_cwt)
    cwt = list(zip(cwt))[0][0][1]
    cwt = round(cwt, 2)

    return peak, abs_sum, diff_statis, auto_co, c3, adf, complexity, line, bin_entropy, appro_entropy, fly, fly_change, cwt

def RQA(ts):
    rp = RecurrencePlot(np.array(ts), dim=1, tau=0, metric="supremum",
                        normalize=False, recurrence_rate=0.05)
    DET = rp.determinism(l_min=2)
    LAM = rp.laminarity(v_min=2)
    # RR = rp.recurrence_rate()
    # print("Recurrence rate:", RR)
    # print("Determinism:", DET)
    # print("Laminarity:", LAM)
    return DET, LAM


#############################################################
# 数据处理、特征提取函数
#############################################################

def get_features_everday(df_data, label_y):

    df_features = pd.DataFrame()
    # ---------------9个 PSD特征
    delta_list = []
    theta_list = []
    alpha_list = []
    beta_list = []
    gamma_list = []
    beta_l_list = []
    beta_m_list = []
    beta_h_list = []
    psd_total_list = []
    # ---------------9个 DE特征
    de_delta_list = []
    de_theta_list = []
    de_alpha_list = []
    de_beta_list = []
    de_gamma_list = []
    de_beta_l_list = []
    de_beta_m_list = []
    de_beta_h_list = []
    de_total_list = []
    # ---------------5个相对功率谱
    relative_delta_list = []
    relative_theta_list = []
    relative_alpha_list = []
    relative_beta_list = []
    relative_gamma_list = []
    # ---------------6个频带的中心频率
    c_f_delta_list = []
    c_f_theta_list = []
    c_f_alpha_list = []
    c_f_beta_list = []
    c_f_gamma_list = []
    c_f_all_list = []
    # ---------------6个频带的带宽
    bw_delta_list = []
    bw_theta_list = []
    bw_alpha_list = []
    bw_beta_list = []
    bw_gamma_list = []
    bw_all_list = []
    # ---------------3个统计特征
    averagee_list = []
    varr_list = []
    maxx_list = []
    covv_list = []
    # ---------------3个hjorth
    h_activity_list = []
    h_mobility_list = []
    h_complexity_list = []
    # ---------------13个tsfresh提取的统计特征
    peak_list = []
    abs_sum_list = []
    diff_statis_list = []
    auto_co_list = []
    c3_list = []
    adf_list = []
    complexity_list = []
    line_list = []
    bin_entropy_list = []
    appro_entropy_list = []
    fly_list = []
    fly_change_list = []
    cwt_list = []
    # ---------------3熵
    samp_entropy_list = []
    fuzzy_entropy_list = []
    # shannon_entropy_list = []
    # permutation_entropy_list = []
    # ---------------2个RAQ
    DET_list = []
    LAM__list = []

    for eegdata in df_data:

        delta, theta, alpha, beta, gamma, beta_l, beta_m, beta_h, allpsd, \
        de_delta, de_theta, de_alpha, de_beta, de_gamma, de_beta_l, de_beta_m, de_beta_h, de_all= PSDandDE(eegdata)

        central_frequency_delta, central_frequency_theta, central_frequency_alpha, central_frequency_beta,\
        central_frequency_gamma, central_frequency_all, bandwidth_delta, bandwidth_theta, bandwidth_alpha, \
        bandwidth_beta, bandwidth_gamma, bandwidth_all\
            = psd(eegdata)

        averagee, varr, covv, maxx = statisticFeatures(eegdata)

        h_activity, h_mobility, h_complexity = hjorth(eegdata)

        peak, abs_sum, diff_statis, auto_co, c3, adf, complexity, line, bin_entropy, appro_entropy,\
                                                         fly, fly_change, cwt = tsfresh_get_features(eegdata)

        # # 调包的
        # # samp_entropy = sampen2(eegdata)
        # # samp_entropy = samp_entropy[0][1]

        # samp_entropy = SampEn(eegdata, m=2, r=0.2 * np.std(eegdata))
        # fuzzy_entropy = Fuzzy_Entropy(eegdata, 5, r=0.25, n=2)

        # permutation_entropy = pec(eegdata, 5, 1)

        # shannon_entropy = S_entropy(eegdata)

        DET, LAM = RQA(eegdata)


        # ---------------
        delta_list.append(delta)
        theta_list.append(theta)
        alpha_list.append(alpha)
        beta_list.append(beta)
        gamma_list.append(gamma)
        beta_l_list.append(beta_l)
        beta_m_list.append(beta_m)
        beta_h_list.append(beta_h)
        psd_total_list.append(allpsd)
        # ---------------
        de_delta_list.append(de_delta)
        de_theta_list.append(de_theta)
        de_alpha_list.append(de_alpha)
        de_beta_list.append(de_beta)
        de_gamma_list.append(de_gamma)
        de_beta_l_list.append(de_beta_l)
        de_beta_m_list.append(de_beta_m)
        de_beta_h_list.append(de_beta_h)
        de_total_list.append(de_all)
        # ---------------
        relative_delta = delta / allpsd
        relative_theta = theta / allpsd
        relative_alpha = alpha / allpsd
        relative_beta = beta / allpsd
        relative_gamma = gamma / allpsd
        relative_delta_list.append(relative_delta)
        relative_theta_list.append(relative_theta)
        relative_alpha_list.append(relative_alpha)
        relative_beta_list.append(relative_beta)
        relative_gamma_list.append(relative_gamma)
        # ---------------
        c_f_delta_list.append(central_frequency_delta)
        c_f_theta_list.append(central_frequency_theta)
        c_f_alpha_list.append(central_frequency_alpha)
        c_f_beta_list.append(central_frequency_beta)
        c_f_gamma_list.append(central_frequency_gamma)
        c_f_all_list.append(central_frequency_all)
        # ---------------
        bw_delta_list.append(bandwidth_delta)
        bw_theta_list.append(bandwidth_theta)
        bw_alpha_list.append(bandwidth_alpha)
        bw_beta_list.append(bandwidth_beta)
        bw_gamma_list.append(bandwidth_gamma)
        bw_all_list.append(bandwidth_all)
        # ---------------
        averagee_list.append(averagee)
        varr_list.append(varr)
        covv_list.append(covv)
        maxx_list.append(maxx)
        # ---------------
        h_activity_list.append(h_activity)
        h_mobility_list.append(h_mobility)
        h_complexity_list.append(h_complexity)
        # ---------------
        peak_list.append(peak)
        abs_sum_list.append(abs_sum)
        diff_statis_list.append(diff_statis)
        auto_co_list.append(auto_co)
        c3_list.append(c3)
        adf_list.append(adf)
        complexity_list.append(complexity)
        line_list.append(line)
        bin_entropy_list.append(bin_entropy)
        appro_entropy_list.append(appro_entropy)
        fly_list.append(fly)
        fly_change_list.append(fly_change)
        cwt_list.append(cwt)
        # ---------------
        # samp_entropy_list.append(samp_entropy)
        # fuzzy_entropy_list.append(fuzzy_entropy)
        # shannon_entropy_list.append(shannon_entropy)
        # permutation_entropy_list.append(permutation_entropy)
        # ---------------2个RAQ
        DET_list.append(DET)
        LAM__list.append(LAM)

    # 功率谱： 9
    df_features['delta'] = delta_list
    df_features['theta'] = theta_list
    df_features['alpha'] = alpha_list
    df_features['beta'] = beta_list
    df_features['gamma'] = gamma_list
    df_features['beta_l'] = beta_l_list
    df_features['beta_m'] = beta_m_list
    df_features['beta_h'] = beta_h_list
    df_features['total_psd'] = psd_total_list
    # DE： 9
    df_features['de_delta'] = de_delta_list
    df_features['de_theta'] = de_theta_list
    df_features['de_alpha'] = de_alpha_list
    df_features['de_beta'] = de_beta_list
    df_features['de_gamma'] = de_gamma_list
    df_features['de_beta_l'] = de_beta_l_list
    df_features['de_beta_m'] = de_beta_m_list
    df_features['de_beta_h'] = de_beta_h_list
    df_features['de_total_psd'] = de_total_list
    # 相对功率谱
    df_features['re_delta'] = relative_delta_list
    df_features['re_theta'] = relative_theta_list
    df_features['re_alpha'] = relative_alpha_list
    df_features['re_beta'] = relative_beta_list
    df_features['re_gamma'] = relative_gamma_list
    # 中心频率
    df_features['c_delta'] = c_f_delta_list
    df_features['c_theta'] = c_f_theta_list
    df_features['c_alpha'] = c_f_alpha_list
    df_features['c_beta'] = c_f_beta_list
    df_features['c_gamma'] = c_f_gamma_list
    df_features['c_allband'] = c_f_all_list
    # 频带的带宽
    df_features['bw_delta'] = bw_delta_list
    df_features['bw_theta'] = bw_theta_list
    df_features['bw_alpha'] = bw_alpha_list
    df_features['bw_beta'] = bw_beta_list
    df_features['bw_gamma'] = bw_gamma_list
    df_features['bw_allband'] = bw_all_list
    # 统计特征
    df_features['sta_mean'] = averagee_list
    df_features['sta_var'] = varr_list
    df_features['sta_cov'] = covv_list
    df_features['sta_max'] = maxx_list
    # hjorth：3
    df_features['hjorth_activity'] = h_activity_list
    df_features['hjorth_mobility'] = h_mobility_list
    df_features['hjorth_complexity'] = h_complexity_list
    # 统计特征列名： 11
    df_features['peak'] = peak_list
    df_features['abs_sum'] = abs_sum_list
    df_features['diff_statis'] = diff_statis_list
    df_features['auto_co'] = auto_co_list
    df_features['c3'] = c3_list
    df_features['adf'] = adf_list
    df_features['complexity'] = complexity_list
    df_features['line'] = line_list
    df_features['fly'] = fly_list
    df_features['fly_change'] = fly_change_list
    df_features['cwt'] = cwt_list
    # 信息熵特征：数据片段相似性  3 个
    df_features['bin_entropy'] = bin_entropy_list
    df_features['appro_entropy'] = appro_entropy_list
    # df_features['samp_entropy'] = samp_entropy_list
    # df_features['fuzzy_entropy'] = fuzzy_entropy_list
    # df_features['shannon_entropy'] = shannon_entropy_list
    # df_features['permutation_entropy'] = permutation_entropy_list
    # 复发图参数分析
    df_features['DET'] = DET_list
    df_features['LAM'] = LAM__list

    df_features['y'] = label_y

    return df_features


# 保存数据
def save2h5(data_list, label_list):
    fre_data = np.array(data_list)
    label_np = np.array(label_list)
    print('数据集中原始的标签顺序是:\n', label_np)
    print('数据集中原始数据长度:\n', len(label_np))

    f = h5py.File('Dataset_PSD_train.h5', 'w')
    f['fre'] = fre_data
    f['labels'] = label_np
    f.close()


if __name__ == '__main__':
    # load data
    eegdatas, labels = load_dataset('20210509_Dataset_6000.h5')

    # # for test
    # coca = scio.loadmat('data/coca.mat')  # 5400*7
    # coca_data = coca['coca']
    # alll = coca_data[0]
    # X = np.array(alll[:540000])
    # X = X.reshape(108000, 5)  # 降采样,1000Hz-->200Hz
    # X = pd.DataFrame(X)
    # XX = list(X[0])
    # k = 0
    # data = XX[k:(k + 6000)]
    # labels = 1
    # eegdatas = [data]


    # import time

    # t0 = time.time()

    # get feature, what need to kown is that already merged x with y
    eeg_features = get_features_everday(eegdatas, labels)

    # print(eeg_features.shape)
    # print(time.time()-t0, "seconds process time")

    eeg_features.to_csv("ML_features_20210508_59.csv", index=False)  # , header=False

    # save



