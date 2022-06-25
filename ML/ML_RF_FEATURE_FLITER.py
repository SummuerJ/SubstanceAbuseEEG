# 小白鼠6只为训练集，一只为测试集
# 随机森林
# 准确率达99.2～100%

import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pylab as plt
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
import numpy as np
from pandas import read_hdf
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
import os
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# 1) Load data.
dataset = pd.read_csv("ML_features_20210508_59.csv", header=0, index_col=None)
print(dataset)
featurelist = dataset.columns
featurelist = dataset.columns
# print(featurelist)
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


# test_list = [[0, 9], [9, 12], [12, 14], [14, 16], [16,25], [25, 29], [29, 40], [40, 45], [45, 51], [51, 57]]
test_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 57]

# 2) shuffle
shuffled_data = dataset.reindex(np.random.permutation(dataset.index))
shuffled_label = shuffled_data['y']
shuffled_data = shuffled_data.drop('y', axis=1)



# #########################
# shuffled_data = shuffled_data[featurelist_old]
#
# # 3) split dataset into two parts
# train_x = shuffled_data.values[:3513, :8]  # 23220
# train_y = shuffled_label.values[:3513]
# test_x = shuffled_data.values[3513:, :8]
# test_y = shuffled_label.values[3513:]
#
# # 4) Standardlized the feature
# s = StandardScaler()
# s.fit(train_x)
# train_x = s.transform(train_x)
# test_x = s.transform(test_x)
#
# # 5) check y whether is int
# train_y = np.around(train_y)
# test_y = np.around(test_y)
#
# # 5) check the shape of data
# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
#
# # # 6) model and RandomizedSearchCV
# # param_dist = {"n_estimators": sp_randint(1, 30),
# #               "max_depth": sp_randint(1, 15),
# #               "min_samples_split": sp_randint(1, 20),
# #               "min_samples_leaf": sp_randint(2, 10)}
# #
# # clf = RandomForestClassifier(random_state=25, n_jobs=-1)
# #
# # rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
# #                                n_iter=100, cv=3, scoring='accuracy', random_state=25, return_train_score=True)
# # rf_random.fit(train_x, train_y)
# # print('mean test scores', rf_random.cv_results_['mean_test_score'])  # cv_results_
# # # print(rf_random.cv_results_)
# # print('mean train scores', rf_random.cv_results_['mean_train_score'])
# # # cv_results_：给出不同参数情况下的评价结果的记录
# # print(rf_random.best_estimator_)
# # # print(rf_random.cv_results_.keys())
# # print(rf_random.best_score_)
# # for p, s in zip(rf_random.cv_results_['params'], rf_random.cv_results_['mean_test_score']):
# #     print(p, s)
#
# # 7)best clf
# # clf_best = rf_random.best_estimator_
# clf_best = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                                    criterion='gini', max_depth=10, max_features='auto',
#                                    max_leaf_nodes=None, max_samples=None,
#                                    min_impurity_decrease=0.0, min_impurity_split=None,
#                                    min_samples_leaf=6, min_samples_split=15,
#                                    min_weight_fraction_leaf=0.0, n_estimators=29, n_jobs=-1,
#                                    oob_score=False, random_state=25, verbose=0,
#                                    warm_start=False)
# clf_best.fit(train_x, train_y)
#
# # 8) predict
# y_test_pred = clf_best.predict(test_x)
#
# # 9) metrics
# recall1 = recall_score(test_y, y_test_pred)
# print('Test recall', recall1)
# precision1 = precision_score(test_y, y_test_pred, average='weighted')
# print('Test precision', precision1)
# f1_score1 = f1_score(test_y, y_test_pred)
# print('Test f1 score', f1_score1)
# accuracy1 = accuracy_score(test_y, y_test_pred)
# print('Test auccuracy', accuracy1)
# #########################


BEST_SCORE = []
Best_PARAMS = []
for rankplan in range(5):
    for epochRound in test_list:
        print(epochRound)
        feature_rank = new_rank[rankplan]
        shuffled_data = shuffled_data[feature_rank]

        # 3) split dataset into two parts
        train_x = shuffled_data.values[:3513, :epochRound]  # 23220
        train_y = shuffled_label.values[:3513]
        test_x = shuffled_data.values[3513:, :epochRound]
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

        # # 6) model and RandomizedSearchCV
        # param_dist = {"n_estimators": sp_randint(1, 30),
        #               "max_depth": sp_randint(1, 15),
        #               "min_samples_split": sp_randint(1, 20),
        #               "min_samples_leaf": sp_randint(2, 10)}
        #
        # clf = RandomForestClassifier(random_state=25, n_jobs=-1)
        #
        # rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
        #                                n_iter=100, cv=3, scoring='accuracy', random_state=25, return_train_score=True)
        # rf_random.fit(train_x, train_y)
        # print('mean test scores', rf_random.cv_results_['mean_test_score'])  # cv_results_
        # # print(rf_random.cv_results_)
        # print('mean train scores', rf_random.cv_results_['mean_train_score'])
        # # cv_results_：给出不同参数情况下的评价结果的记录
        # print(rf_random.best_estimator_)
        # # print(rf_random.cv_results_.keys())
        # print(rf_random.best_score_)
        # for p, s in zip(rf_random.cv_results_['params'], rf_random.cv_results_['mean_test_score']):
        #     print(p, s)

        # 7)best clf
        # clf_best = rf_random.best_estimator_
        clf_best = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                           criterion='gini', max_depth=10, max_features='auto',
                                           max_leaf_nodes=None, max_samples=None,
                                           min_impurity_decrease=0.0, min_impurity_split=None,
                                           min_samples_leaf=6, min_samples_split=15,
                                           min_weight_fraction_leaf=0.0, n_estimators=29, n_jobs=-1,
                                           oob_score=False, random_state=25, verbose=0,
                                           warm_start=False)
        clf_best.fit(train_x, train_y)

        # 8) predict
        y_test_pred = clf_best.predict(test_x)

        # 9) metrics
        recall1 = recall_score(test_y, y_test_pred)
        print('Test recall', recall1)
        precision1 = precision_score(test_y, y_test_pred, average='weighted')
        print('Test precision', precision1)
        f1_score1 = f1_score(test_y, y_test_pred)
        print('Test f1 score', f1_score1)
        accuracy1 = accuracy_score(test_y, y_test_pred)
        print('Test auccuracy', accuracy1)

        SCORE = [recall1, precision1, f1_score1, accuracy1, rankplan, epochRound]
        BEST_SCORE.append(SCORE)


print(BEST_SCORE)
BEST_SCORE = pd.DataFrame(BEST_SCORE, columns=['recall', 'precision', 'f1_score', 'accuracy', 'plan', 'fea_num'])
BEST_SCORE.loc['mean'] = BEST_SCORE.apply(lambda x: x.mean())
# Best_PARAMS.append('0')
# BEST_SCORE['best_params'] = Best_PARAMS
# BEST_SCORE['features_num'] = test_list[:][1]
print(BEST_SCORE)
BEST_SCORE.to_csv('RF_59_BEST_SCORE_20210516_featurepick_5.csv')  # , index=False

# features = featurelist[:-1]
# importances = clf_best.feature_importances_
# print(importances)
# indices = (np.argsort(importances))[:]
# plt.figure(figsize=(10, 12))
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='r', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()
# print([features[i] for i in indices])

# # 为每个类别计算ROC曲线和AUC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# fpr, tpr, _ = roc_curve(test_y, y_test_pred)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange',
#          lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


