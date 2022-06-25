import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#%config InlineBackend.figure_format = 'retina'
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import scipy.io as scio
import numpy as np
from sklearn.model_selection import StratifiedKFold #交叉验证
import copy
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# 1) Load data.
dataset = pd.read_csv("ML_features_20210508_59.csv", header=0, index_col=None)
print(dataset)
featurelist = dataset.columns

featurelist_old = ['delta', 'theta', 'alpha', 'beta', 'gamma',  'hjorth_activity', 'hjorth_mobility','hjorth_complexity',
                   'peak', 'abs_sum', 'diff_statis',
                   'auto_co', 'c3', 'beta_l', 'beta_m', 'beta_h', 'total_psd', 'de_delta', 'de_theta', 'de_alpha',
                   'de_beta', 'de_gamma', 'de_beta_l', 'de_beta_m', 'de_beta_h', 'de_total_psd',
                   're_delta', 're_theta', 're_alpha', 're_beta', 're_gamma', 'c_delta',
                   'c_theta', 'c_alpha', 'c_beta', 'c_gamma', 'c_allband', 'bw_delta',
                   'bw_theta', 'bw_alpha', 'bw_beta', 'bw_gamma', 'bw_allband', 'sta_mean',
                   'sta_var', 'sta_cov', 'sta_max',
                   'adf', 'complexity', 'line', 'fly', 'fly_change', 'cwt', 'bin_entropy',
                   'appro_entropy', 'DET', 'LAM']

feature_rank_refer = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'beta_l', 'beta_m', 'beta_h', 'total_psd',
                      'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
                      'bin_entropy', 'appro_entropy',
                      'DET', 'LAM',
                      'de_delta', 'de_theta', 'de_alpha', 'de_beta', 'de_gamma', 'de_beta_l', 'de_beta_m', 'de_beta_h', 'de_total_psd',
                      'sta_mean', 'sta_var', 'sta_cov', 'sta_max',
                      'peak', 'abs_sum', 'diff_statis', 'auto_co', 'c3', 'adf', 'complexity', 'line', 'fly', 'fly_change', 'cwt',
                      're_delta', 're_theta', 're_alpha', 're_beta', 're_gamma',
                      'c_delta', 'c_theta', 'c_alpha', 'c_beta', 'c_gamma', 'c_allband',
                      'bw_delta', 'bw_theta', 'bw_alpha', 'bw_beta', 'bw_gamma', 'bw_allband']
MIQ_LIST = ['abs_sum', 'de_delta', 'bw_alpha', 'bw_allband', 'c_beta', 'complexity', 'bw_beta', 'c_allband', 'c_gamma', 'line', 'sta_max', 'bw_gamma', 'bw_theta', 'c_delta', 'de_alpha', 'peak', 'bw_delta', 'fly_change', 'de_total_psd', 'hjorth_complexity', 'de_beta_l', 'appro_entropy', 'c_theta', 'de_beta_h', 'de_gamma', 'c_alpha', 'de_theta', 'de_beta', 'de_beta_m', 'fly', 'bin_entropy', 're_alpha', 're_beta', 're_delta', 're_gamma', 'beta', 'alpha', 'auto_co', 're_theta', 'total_psd', 'beta_h', 'DET', 'sta_mean', 'sta_cov', 'LAM', 'theta', 'hjorth_mobility', 'sta_var', 'cwt', 'adf', 'hjorth_activity', 'beta_l', 'gamma', 'c3', 'beta_m', 'diff_statis', 'delta']
MID_LIST = ['abs_sum', 're_alpha', 're_beta', 'de_delta', 'bw_alpha', 'bw_allband', 're_delta', 're_gamma', 'c_beta', 'complexity', 'beta', 'alpha', 'auto_co', 're_theta', 'bw_beta', 'total_psd', 'c_allband', 'beta_h', 'DET', 'sta_mean', 'c_gamma', 'line', 'sta_max', 'sta_cov', 'LAM', 'theta', 'bw_gamma', 'hjorth_mobility', 'sta_var', 'cwt', 'adf', 'hjorth_activity', 'beta_l', 'gamma', 'c3', 'beta_m', 'bw_theta', 'c_delta', 'diff_statis', 'delta', 'peak', 'de_alpha', 'bw_delta', 'bin_entropy', 'c_theta', 'appro_entropy', 'de_beta_l', 'hjorth_complexity', 'fly_change', 'c_alpha', 'de_total_psd', 'fly', 'de_beta_h', 'de_theta', 'de_beta_m', 'de_gamma', 'de_beta']
feature_importances = ['DET', 'complexity', 'abs_sum', 'de_beta_l', 'appro_entropy', 'gamma', 'LAM', 're_delta', 'delta', 'diff_statis', 'beta_l', 're_theta', 'hjorth_complexity', 'de_beta_m', 'theta', 'hjorth_mobility', 'c_allband', 're_gamma', 'alpha', 'de_gamma', 'de_theta', 'beta_m', 'c_beta', 'de_alpha', 're_alpha', 'de_beta_h', 'de_beta', 'sta_var', 'de_delta', 'sta_cov', 'beta', 'de_total_psd', 'auto_co', 'total_psd', 'bw_allband', 'sta_max', 'peak', 're_beta', 'bw_alpha', 'bw_theta', 'hjorth_activity', 'beta_h', 'c_alpha', 'c_theta', 'sta_mean', 'fly', 'bw_beta', 'line', 'c_delta', 'bin_entropy', 'bw_gamma', 'bw_delta', 'c_gamma', 'cwt', 'fly_change', 'c3', 'adf']
MIC = ['DET', 'LAM', 'abs_sum', 'adf', 'alpha', 'appro_entropy', 'auto_co', 'beta', 'beta_h', 'beta_l', 'beta_m', 'bin_entropy', 'bw_allband', 'bw_alpha', 'bw_beta', 'bw_delta', 'bw_gamma', 'bw_theta', 'c3', 'c_allband', 'c_alpha', 'c_beta', 'c_delta', 'c_gamma', 'c_theta', 'complexity', 'cwt', 'de_alpha', 'de_beta', 'de_beta_h', 'de_beta_l', 'de_beta_m', 'de_delta', 'de_gamma', 'de_theta', 'de_total_psd', 'delta', 'diff_statis', 'fly', 'fly_change', 'gamma', 'hjorth_activity', 'hjorth_complexity', 'hjorth_mobility', 'line', 'peak', 're_alpha', 're_beta', 're_delta', 're_gamma', 're_theta', 'sta_cov', 'sta_max', 'sta_mean', 'sta_var', 'theta', 'total_psd']

new_rank = [feature_rank_refer, MIQ_LIST, MID_LIST, feature_importances, MIC]
# print(dataset.columns)
test_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 57]

# 2) shuffle
shuffled_data = dataset.reindex(np.random.permutation(dataset.index))
shuffled_label = shuffled_data['y']
shuffled_data = shuffled_data.drop('y', axis=1)



####################
dataset = dataset[featurelist_old]

# 3) split dataset into two parts
train_x = shuffled_data.values[:3513, :]  # 23220
train_y = shuffled_label.values[:3513]
test_x = shuffled_data.values[3513:, :]
test_y = shuffled_label.values[3513:]

# 4) Standardlized the feature
s = StandardScaler()
s.fit(train_x)
train_x = s.transform(train_x)
test_x = s.transform(test_x)

# 5) check y whether is int
train_y = np.around(train_y)
test_y = np.around(test_y)

# 5) check the shape of data
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

knn_clf = KNeighborsClassifier()
# 使用网格搜索的方法进行对超参数的选取
# 超参数是指在算法进行时对不确定的参数进行调整以使得算法的预测准确率最高
params = [
    {"weights": ["uniform"],
     "n_neighbors": [k for k in range(1, 11)]
     },
    {"weights": ["distance"],
     "n_neighbors": [k for k in range(1, 11)],
     "p": [k for k in range(1, 6)]
     }
]

# n_jobs 这个参数是使用计算机的几个核，赋值为-1表示全都使用

gridsearch = GridSearchCV(knn_clf, params, n_jobs=-1)
gridsearch.fit(train_x, train_y)
print("best_score", gridsearch.best_score_)
print("best_params", gridsearch.best_params_)

for p, s in zip(gridsearch.cv_results_['params'], gridsearch.cv_results_['mean_test_score']):
    print(p, s)


# print(knn_clf.score(test_x, test_y))

results_test = knn_clf.predict(test_x)

recall1 = recall_score(test_y, results_test)
print('Test recall', recall1)

precision1 = precision_score(test_y, results_test, average='weighted')
print('Test precision', precision1)

f1_score1 = f1_score(test_y, results_test)
print('Test f1 score', f1_score1)

accuracy1 = accuracy_score(test_y, results_test)
print('Test auccuracy', accuracy1)
####################




# BEST_SCORE = []
# Best_PARAMS = []
# for rankplan in range(5):
#     for epochRound in test_list:
#         print(epochRound)
#         feature_rank = new_rank[rankplan]
#         shuffled_data = shuffled_data[feature_rank]
#
#         # 3) split dataset into two parts
#         train_x = shuffled_data.values[:3513, :epochRound]  # 23220
#         train_y = shuffled_label.values[:3513]
#         test_x = shuffled_data.values[3513:, :epochRound]
#         test_y = shuffled_label.values[3513:]
#
#         # 4) Standardlized the feature
#         s = StandardScaler()
#         s.fit(train_x)
#         train_x = s.transform(train_x)
#         test_x = s.transform(test_x)
#
#         # 5) check y whether is int
#         train_y = np.around(train_y)
#         test_y = np.around(test_y)
#
#         # 5) check the shape of data
#         print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
#
#         # knn_clf = KNeighborsClassifier()
#         # # 使用网格搜索的方法进行对超参数的选取
#         # # 超参数是指在算法进行时对不确定的参数进行调整以使得算法的预测准确率最高
#         # params = [
#         #     {"weights": ["uniform"],
#         #      "n_neighbors": [k for k in range(1, 11)]
#         #      },
#         #     {"weights": ["distance"],
#         #      "n_neighbors": [k for k in range(1, 11)],
#         #      "p": [k for k in range(1, 6)]
#         #      }
#         # ]
#         #
#         # # n_jobs 这个参数是使用计算机的几个核，赋值为-1表示全都使用
#         #
#         # gridsearch = GridSearchCV(knn_clf, params, n_jobs=-1)
#         # gridsearch.fit(train_x, train_y)
#         # print("best_score", gridsearch.best_score_)
#         # print("best_params", gridsearch.best_params_)
#         #
#         # for p, s in zip(gridsearch.cv_results_['params'], gridsearch.cv_results_['mean_test_score']):
#         #     print(p, s)
#
#         # 将最好的赋值给knn_clf
#         # knn_clf = gridsearch.best_estimator_
#         # print(knn_clf)
#         knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                                          metric_params=None, n_jobs=None, n_neighbors=10, p=1,
#                                          weights='distance')
#         knn_clf.fit(train_x, train_y)
#
#         # print(knn_clf.score(test_x, test_y))
#
#         results_test = knn_clf.predict(test_x)
#
#         recall1 = recall_score(test_y, results_test)
#         print('Test recall', recall1)
#
#         precision1 = precision_score(test_y, results_test, average='weighted')
#         print('Test precision', precision1)
#
#         f1_score1 = f1_score(test_y, results_test)
#         print('Test f1 score', f1_score1)
#
#         accuracy1 = accuracy_score(test_y, results_test)
#         print('Test auccuracy', accuracy1)
#
#         SCORE = [recall1, precision1, f1_score1, accuracy1, rankplan, epochRound]
#         BEST_SCORE.append(SCORE)
#
#
# print(BEST_SCORE)
# BEST_SCORE = pd.DataFrame(BEST_SCORE, columns=['recall', 'precision', 'f1_score', 'accuracy', 'plan', 'fea_num'])
# BEST_SCORE.loc['mean'] = BEST_SCORE.apply(lambda x: x.mean())
# # Best_PARAMS.append('0')
# # BEST_SCORE['best_params'] = Best_PARAMS
# # BEST_SCORE['features_num'] = test_list[:][1]
# print(BEST_SCORE)
# BEST_SCORE.to_csv('KNN_59_BEST_SCORE_20210516_featurepick_5.csv')  # , index=False
