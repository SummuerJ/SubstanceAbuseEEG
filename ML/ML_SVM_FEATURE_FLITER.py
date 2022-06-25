import pickle
from sklearn.model_selection import GridSearchCV
#%config InlineBackend.figure_format = 'retina'
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import StratifiedKFold #交叉验证
import copy
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# 4)对网格搜索结果画热图     找到验证集上最好的C
def draw_heatmap_linear(acc, acc_desc, c_list, num):
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(acc, annot=True, fmt='.3f', yticklabels=c_list, xticklabels=[])
    ax.collections[0].colorbar.set_label("accuracy")
    ax.set(ylabel='$C$')
    plt.title(acc_desc + ' w.r.t $C$')
    sns.set_style("whitegrid", {'axes.grid': False})
    plt.savefig('20210514svmoldfliter/20210514_svm_'+acc_desc+'_'+str(num)+'.png')
    # plt.show()


dataset = pd.read_csv("ML_features_20210508_59.csv", header=0, index_col=None)
print(dataset)
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

# shuffle
shuffled_data = dataset.reindex(np.random.permutation(dataset.index))
# shuffled_data = shuffled_data[feature_rank_refer]
shuffled_label = shuffled_data['y']
shuffled_data = shuffled_data.drop('y', axis=1)

# test_list = [[0, 9], [9, 12], [12, 14], [14, 16], [16,25], [25, 29], [29, 40], [40, 45], [45, 51], [51, 57]]
test_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 57]


# ##################
# shuffled_data = shuffled_data[featurelist_old]
#
#
# # first = test_list[epochRound][0]
# # rear = test_list[epochRound]
#
# # split dataset
# train_x = shuffled_data.values[:3513, :8]  # 23220
# train_y = shuffled_label.values[:3513]
# test_x = shuffled_data.values[3513:, :8]
# test_y = shuffled_label.values[3513:]
#
# s = StandardScaler()
# s.fit(train_x)
# train_x = s.transform(train_x)
# test_x = s.transform(test_x)
#
# train_y = np.around(train_y)
# test_y = np.around(test_y)
#
# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
#
# # # 3) 使用线性核. 网格搜索找最佳的参数C
# # classifier = svm.SVC()  # probability=True
# # C_list = np.logspace(-6, -1, 6)  # [1.e-06 1.e-05 1.e-04 1.e-03 1.e-02 1.e-01]
# # param_grid = [{'kernel': ['rbf'], 'C': [1],'gamma': [0.1]}]
# # params = [
# #     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
# #     {'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3]},
# #     {'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
# #      'gamma': [1, 0.1, 0.01, 0.001]}]
#
# # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]  # 学习率
# # gamma = [1, 0.1, 0.01, 0.001]
# # param_grid = dict(learning_rate=learning_rate, gamma=gamma)  # 转化为字典格式，网络搜索要求
# #
# # kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)  # 将训练/测试数据集划分10个互斥子集，
#
# # # Cross Validation
# # grid = ms.GridSearchCV(classifier, param_grid=param_grid, cv=3, return_train_score=True)
#
# # grid.fit(train_x, train_y)
# # print("The best parameters C are %s with score of %0.2f" % (grid.best_params_, grid.best_score_))
# #
# # mean_train_score0 = grid.cv_results_['mean_train_score']
# # print("The mean train score are %s", mean_train_score0)
# # mean_test_score0 = grid.cv_results_['mean_test_score']
# # print("The mean test score are %s", mean_test_score0)
# # print(mean_train_score0.shape)
# # print(mean_test_score0.shape)
# # mean_train_score0 = np.array([mean_train_score0])
# # mean_train_score0 = mean_train_score0.T
# # mean_test_score0 = np.array([mean_test_score0])
# # mean_test_score0 = mean_test_score0.T
# #
# # train_acc = mean_train_score0
# # draw_heatmap_linear(train_acc, 'train accuracy', grid.cv_results_['params'], rear)
# #
# # val_acc = mean_test_score0
# # draw_heatmap_linear(val_acc, 'val accuracy', grid.cv_results_['params'], rear)
# #
# #
# #
# # print("模型的最优参数：", grid.best_params_)
# # Best_PARAMS.append(grid.best_params_)
# # print("最优模型分数：", grid.best_score_)
# # print("最优模型对象：", grid.best_estimator_)
# # # for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
# # #     print(p, s)
#
#
# # 5) Use the best C to calculate the test accuracy.
# grid = svm.SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#                 decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#                 max_iter=-1, probability=False, random_state=None, shrinking=True,
#                 tol=0.001, verbose=False)
# grid.fit(train_x, train_y)
# results_test = grid.predict(test_x)
#
# recall1 = recall_score(test_y, results_test)
# print('Test recall', recall1)
#
# precision1 = precision_score(test_y, results_test, average='weighted')
# print('Test precision', precision1)
#
# f1_score1 = f1_score(test_y, results_test)
# print('Test f1 score', f1_score1)
#
# accuracy1 = accuracy_score(test_y, results_test)
# print('Test auccuracy', accuracy1)
# ###################


BEST_SCORE = []
Best_PARAMS = []
for rankplan in range(5):
    for epochRound in test_list:
        print(epochRound)
        feature_rank = new_rank[rankplan]
        shuffled_data = shuffled_data[feature_rank]


        # first = test_list[epochRound][0]
        # rear = test_list[epochRound]

        # split dataset
        train_x = shuffled_data.values[:3513, :epochRound]  # 23220
        train_y = shuffled_label.values[:3513]
        test_x = shuffled_data.values[3513:, :epochRound]
        test_y = shuffled_label.values[3513:]

        s = StandardScaler()
        s.fit(train_x)
        train_x = s.transform(train_x)
        test_x = s.transform(test_x)

        train_y = np.around(train_y)
        test_y = np.around(test_y)

        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        # # 3) 使用线性核. 网格搜索找最佳的参数C
        # classifier = svm.SVC()  # probability=True
        # C_list = np.logspace(-6, -1, 6)  # [1.e-06 1.e-05 1.e-04 1.e-03 1.e-02 1.e-01]
        # param_grid = [{'kernel': ['rbf'], 'C': [1],'gamma': [0.1]}]
        # params = [
        #     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
        #     {'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3]},
        #     {'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
        #      'gamma': [1, 0.1, 0.01, 0.001]}]

        # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]  # 学习率
        # gamma = [1, 0.1, 0.01, 0.001]
        # param_grid = dict(learning_rate=learning_rate, gamma=gamma)  # 转化为字典格式，网络搜索要求
        #
        # kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)  # 将训练/测试数据集划分10个互斥子集，

        # # Cross Validation
        # grid = ms.GridSearchCV(classifier, param_grid=param_grid, cv=3, return_train_score=True)

        # grid.fit(train_x, train_y)
        # print("The best parameters C are %s with score of %0.2f" % (grid.best_params_, grid.best_score_))
        #
        # mean_train_score0 = grid.cv_results_['mean_train_score']
        # print("The mean train score are %s", mean_train_score0)
        # mean_test_score0 = grid.cv_results_['mean_test_score']
        # print("The mean test score are %s", mean_test_score0)
        # print(mean_train_score0.shape)
        # print(mean_test_score0.shape)
        # mean_train_score0 = np.array([mean_train_score0])
        # mean_train_score0 = mean_train_score0.T
        # mean_test_score0 = np.array([mean_test_score0])
        # mean_test_score0 = mean_test_score0.T
        #
        # train_acc = mean_train_score0
        # draw_heatmap_linear(train_acc, 'train accuracy', grid.cv_results_['params'], rear)
        #
        # val_acc = mean_test_score0
        # draw_heatmap_linear(val_acc, 'val accuracy', grid.cv_results_['params'], rear)
        #
        #
        #
        # print("模型的最优参数：", grid.best_params_)
        # Best_PARAMS.append(grid.best_params_)
        # print("最优模型分数：", grid.best_score_)
        # print("最优模型对象：", grid.best_estimator_)
        # # for p, s in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        # #     print(p, s)


        # 5) Use the best C to calculate the test accuracy.
        grid = svm.SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
                        max_iter=-1, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False)
        grid.fit(train_x, train_y)
        results_test = grid.predict(test_x)

        recall1 = recall_score(test_y, results_test)
        print('Test recall', recall1)

        precision1 = precision_score(test_y, results_test, average='weighted')
        print('Test precision', precision1)

        f1_score1 = f1_score(test_y, results_test)
        print('Test f1 score', f1_score1)

        accuracy1 = accuracy_score(test_y, results_test)
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
BEST_SCORE.to_csv('SVM_59_BEST_SCORE_20210516_featurepick_5.csv')  # , index=False
