import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pymrmr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from minepy import MINE


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


dataset = pd.read_csv("ML_features_20210508_59.csv", header=0, index_col=None)
print(dataset)
featurelist = dataset.columns
# print(featurelist)
featurelist_old = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'beta_l', 'beta_m',
                   'beta_h', 'total_psd', 'de_delta', 'de_theta', 'de_alpha', 'de_beta',
                   'de_gamma', 'de_beta_l', 'de_beta_m', 'de_beta_h', 'de_total_psd',
                   're_delta', 're_theta', 're_alpha', 're_beta', 're_gamma', 'c_delta',
                   'c_theta', 'c_alpha', 'c_beta', 'c_gamma', 'c_allband', 'bw_delta',
                   'bw_theta', 'bw_alpha', 'bw_beta', 'bw_gamma', 'bw_allband', 'sta_mean',
                   'sta_var', 'sta_cov', 'sta_max', 'hjorth_activity', 'hjorth_mobility',
                   'hjorth_complexity', 'peak', 'abs_sum', 'diff_statis', 'auto_co', 'c3',
                   'adf', 'complexity', 'line', 'fly', 'fly_change', 'cwt', 'bin_entropy',
                   'appro_entropy', 'DET', 'LAM', 'y']

feature_rank_refer = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'beta_l', 'beta_m', 'beta_h', 'total_psd',
                      'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
                      'bin_entropy', 'appro_entropy',
                      'DET', 'LAM',
                      'de_delta', 'de_theta', 'de_alpha', 'de_beta', 'de_gamma', 'de_beta_l', 'de_beta_m', 'de_beta_h', 'de_total_psd',
                      'sta_mean', 'sta_var', 'sta_cov', 'sta_max',
                      'peak', 'abs_sum', 'diff_statis', 'auto_co', 'c3', 'adf', 'complexity', 'line', 'fly', 'fly_change', 'cwt',
                      're_delta', 're_theta', 're_alpha', 're_beta', 're_gamma',
                      'c_delta', 'c_theta', 'c_alpha', 'c_beta', 'c_gamma', 'c_allband',
                      'bw_delta', 'bw_theta', 'bw_alpha', 'bw_beta', 'bw_gamma', 'bw_allband',
                      'y']
dataset = dataset[feature_rank_refer]
print(dataset.columns)

test_list = list(i for i in range(57))

# shuffle
shuffled_data = dataset.reindex(np.random.permutation(dataset.index))

feature_rank_refer = ['y', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'beta_l', 'beta_m', 'beta_h', 'total_psd',
                      'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
                      'bin_entropy', 'appro_entropy',
                      'DET', 'LAM',
                      'de_delta', 'de_theta', 'de_alpha', 'de_beta', 'de_gamma', 'de_beta_l', 'de_beta_m', 'de_beta_h', 'de_total_psd',
                      'sta_mean', 'sta_var', 'sta_cov', 'sta_max',
                      'peak', 'abs_sum', 'diff_statis', 'auto_co', 'c3', 'adf', 'complexity', 'line', 'fly', 'fly_change', 'cwt',
                      're_delta', 're_theta', 're_alpha', 're_beta', 're_gamma',
                      'c_delta', 'c_theta', 'c_alpha', 'c_beta', 'c_gamma', 'c_allband',
                      'bw_delta', 'bw_theta', 'bw_alpha', 'bw_beta', 'bw_gamma', 'bw_allband'
                      ]
shuffled_data = shuffled_data[feature_rank_refer]

# train_x = shuffled_data.values[:3513, :rear]  # 23220
# train_y = shuffled_label.values[:3513]
# test_x = shuffled_data.values[3513:, :rear]
# test_y = shuffled_label.values[3513:]


ranks = pd.DataFrame()
print('******************MIQ_LIST**********************')
MIQ_LIST = pymrmr.mRMR(shuffled_data, 'MIQ', 57)
print(len(MIQ_LIST))
# ['abs_sum', 'de_delta', 'bw_alpha', 'bw_allband', 'c_beta', 'complexity', 'bw_beta', 'c_allband', 'c_gamma', 'line', 'sta_max', 'bw_gamma', 'bw_theta', 'c_delta', 'de_alpha', 'peak', 'bw_delta', 'fly_change', 'de_total_psd', 'hjorth_complexity', 'de_beta_l', 'appro_entropy', 'c_theta', 'de_beta_h', 'de_gamma', 'c_alpha', 'de_theta', 'de_beta', 'de_beta_m', 'fly', 'bin_entropy', 're_alpha', 're_beta', 're_delta', 're_gamma', 'beta', 'alpha', 'auto_co', 're_theta', 'total_psd', 'beta_h', 'DET', 'sta_mean', 'sta_cov', 'LAM', 'theta', 'hjorth_mobility', 'sta_var', 'cwt', 'adf', 'hjorth_activity', 'beta_l', 'gamma', 'c3', 'beta_m', 'diff_statis', 'delta']

print('******************MID_LIST**********************')
MID_LIST = pymrmr.mRMR(shuffled_data, 'MID', 57)
print(len(MID_LIST))
# ['abs_sum', 're_alpha', 're_beta', 'de_delta', 'bw_alpha', 'bw_allband', 're_delta', 're_gamma', 'c_beta', 'complexity', 'beta', 'alpha', 'auto_co', 're_theta', 'bw_beta', 'total_psd', 'c_allband', 'beta_h', 'DET', 'sta_mean', 'c_gamma', 'line', 'sta_max', 'sta_cov', 'LAM', 'theta', 'bw_gamma', 'hjorth_mobility', 'sta_var', 'cwt', 'adf', 'hjorth_activity', 'beta_l', 'gamma', 'c3', 'beta_m', 'bw_theta', 'c_delta', 'diff_statis', 'delta', 'peak', 'de_alpha', 'bw_delta', 'bin_entropy', 'c_theta', 'appro_entropy', 'de_beta_l', 'hjorth_complexity', 'fly_change', 'c_alpha', 'de_total_psd', 'fly', 'de_beta_h', 'de_theta', 'de_beta_m', 'de_gamma', 'de_beta']



shuffled_label = shuffled_data['y']
shuffled_data = shuffled_data.drop('y', axis=1)

# feature
print('******************Tree_LIST**********************')
feature_importances = ['adf', 'c3', 'fly_change', 'cwt', 'c_gamma', 'bw_delta', 'bw_gamma', 'bin_entropy', 'c_delta', 'line', 'bw_beta', 'fly', 'sta_mean', 'c_theta', 'c_alpha', 'beta_h', 'hjorth_activity', 'bw_theta', 'bw_alpha', 're_beta', 'peak', 'sta_max', 'bw_allband', 'total_psd', 'auto_co', 'de_total_psd', 'beta', 'sta_cov', 'de_delta', 'sta_var', 'de_beta', 'de_beta_h', 're_alpha', 'de_alpha', 'c_beta', 'beta_m', 'de_theta', 'de_gamma', 'alpha', 're_gamma', 'c_allband', 'hjorth_mobility', 'theta', 'de_beta_m', 'hjorth_complexity', 're_theta', 'beta_l', 'diff_statis', 'delta', 're_delta', 'LAM', 'gamma', 'appro_entropy', 'de_beta_l', 'abs_sum', 'complexity', 'DET']
feature_importances.reverse()
print("feature_importances", feature_importances)

# ['DET', 'complexity', 'abs_sum', 'de_beta_l', 'appro_entropy', 'gamma', 'LAM', 're_delta', 'delta', 'diff_statis', 'beta_l', 're_theta', 'hjorth_complexity', 'de_beta_m', 'theta', 'hjorth_mobility', 'c_allband', 're_gamma', 'alpha', 'de_gamma', 'de_theta', 'beta_m', 'c_beta', 'de_alpha', 're_alpha', 'de_beta_h', 'de_beta', 'sta_var', 'de_delta', 'sta_cov', 'beta', 'de_total_psd', 'auto_co', 'total_psd', 'bw_allband', 'sta_max', 'peak', 're_beta', 'bw_alpha', 'bw_theta', 'hjorth_activity', 'beta_h', 'c_alpha', 'c_theta', 'sta_mean', 'fly', 'bw_beta', 'line', 'c_delta', 'bin_entropy', 'bw_gamma', 'bw_delta', 'c_gamma', 'cwt', 'fly_change', 'c3', 'adf']


# MIC
print('******************MIC_LIST**********************')
mine = MINE()
mic_scores = []
# print(shuffled_data.shape[1])
shuffled_data = shuffled_data.values
# print(shuffled_data[:, 1])
for i in range(shuffled_data.shape[1]):
    mine.compute_score(shuffled_data[:, i], shuffled_label)
    m = mine.mic()
    mic_scores.append(m)
rank = rank_to_dict(mic_scores, feature_rank_refer)
rank = sorted(rank.keys())
print(rank)

# ['DET', 'LAM', 'abs_sum', 'adf', 'alpha', 'appro_entropy', 'auto_co', 'beta', 'beta_h', 'beta_l', 'beta_m', 'bin_entropy', 'bw_allband', 'bw_alpha', 'bw_beta', 'bw_delta', 'bw_gamma', 'bw_theta', 'c3', 'c_allband', 'c_alpha', 'c_beta', 'c_delta', 'c_gamma', 'c_theta', 'complexity', 'cwt', 'de_alpha', 'de_beta', 'de_beta_h', 'de_beta_l', 'de_beta_m', 'de_delta', 'de_gamma', 'de_theta', 'de_total_psd', 'delta', 'diff_statis', 'fly', 'fly_change', 'gamma', 'hjorth_activity', 'hjorth_complexity', 'hjorth_mobility', 'line', 'peak', 're_alpha', 're_beta', 're_delta', 're_gamma', 're_theta', 'sta_cov', 'sta_max', 'sta_mean', 'sta_var', 'theta', 'total_psd']


# # 信息贡献度 互信息
# print("信息贡献度 互信息")
# rankk = mutual_info_classif(shuffled_data, shuffled_label)
# rank = rank_to_dict(rankk, feature_rank_refer, order=1)
# rank = sorted(rank.keys())
# print(rank)
#
# # ['DET', 'LAM', 'abs_sum', 'adf', 'alpha', 'appro_entropy', 'auto_co', 'beta', 'beta_h', 'beta_l', 'beta_m', 'bin_entropy', 'bw_allband', 'bw_alpha', 'bw_beta', 'bw_delta', 'bw_gamma', 'bw_theta', 'c3', 'c_allband', 'c_alpha', 'c_beta', 'c_delta', 'c_gamma', 'c_theta', 'complexity', 'cwt', 'de_alpha', 'de_beta', 'de_beta_h', 'de_beta_l', 'de_beta_m', 'de_delta', 'de_gamma', 'de_theta', 'de_total_psd', 'delta', 'diff_statis', 'fly', 'fly_change', 'gamma', 'hjorth_activity', 'hjorth_complexity', 'hjorth_mobility', 'line', 'peak', 're_alpha', 're_beta', 're_delta', 're_gamma', 're_theta', 'sta_cov', 'sta_max', 'sta_mean', 'sta_var', 'theta', 'total_psd']


# from sklearn import svm
# grid = svm.SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#                     decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#                     max_iter=-1, probability=False, random_state=None, shrinking=True,
#                     tol=0.001, verbose=False)
#
# rfe = RFE(grid, n_features_to_select=57)
# rfe.fit(shuffled_data, shuffled_label)
# print(rfe.ranking_)





