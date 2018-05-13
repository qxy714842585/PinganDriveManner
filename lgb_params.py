# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 18:31
# @Author  : qxy
# @File    : main_xgb.py.py
# @Software: PyCharm


import lightgbm as lgb
import pandas as pd

from feature import form_dataset

# path_train = '../PINGAN-2018-train_demo.csv'
path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件


feature = ['num_of_records', 'num_of_trips', 'num_of_state_0', 'num_of_state_1', 'num_of_state_2', 'num_of_state_3',
           'num_of_state_4', 'mean_speed', 'var_speed', 'mean_height', 'var_height', 'tp0', 'tp1', 'tp2', 'tp3', 'tp4',
           'tp5', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'loc0', 'loc1', 'loc2', 'duration', 'turn_n0',
           'turn_n1', 'turn_n2', 'turn_n3']
#, 'sf0', 'sf1', 'sf2', 'sf3'


def read_csv(path):
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):
    data = pd.read_csv(path)
    try:
        data.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE"
            , "DIRECTION", "HEIGHT", "SPEED", "CALLSTATE", "Y"]
    except:
        data.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                       "CALLSTATE"]
    return data


def get_params(params, best_params):
    for i in best_params:
        params[i] = best_params[i]
    return params


train = read_csv(path_train)
train_set = form_dataset(train)

x = train_set[feature].fillna(-1)
train_set['y'] = train_set['target'].apply(lambda x: 0 if x==0 else 1)
label = train_set['y']
lgb_train = lgb.Dataset(x, label, free_raw_data=False)

params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'l2'}

min_merror = float('Inf')
best_params = {}

print('Params_1')
for num_leaves in range(20, 200, 5):
    for max_depth in range(3, 8, 1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth
        cv_results = lgb.cv(params, lgb_train, seed=2018, nfold=3, metrics=['mse'], early_stopping_rounds=10,verbose_eval=None)
        mean_merror = pd.Series(cv_results['l2-mean']).min()
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth

# params['num_leaves'] = best_params['num_leaves']
# params['max_depth'] = best_params['max_depth']
params = get_params(params, best_params)

# print("调参2：降低过拟合")
print('Params_2')
for max_bin in range(1, 255, 5):
    for min_data_in_leaf in range(10, 200, 5):
        params['max_bin'] = max_bin
        params['min_data_in_leaf'] = min_data_in_leaf
        cv_results = lgb.cv(params, lgb_train, seed=42, nfold=3, metrics=['mse'], early_stopping_rounds=3,verbose_eval=None)

        mean_merror = pd.Series(cv_results['l2-mean']).min()
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['max_bin'] = max_bin
            best_params['min_data_in_leaf'] = min_data_in_leaf

# params['max_bin'] = best_params['max_bin']
# params['min_data_in_leaf'] = best_params['min_data_in_leaf']
params = get_params(params, best_params)

# print("调参3：降低过拟合")
print('Params_3')
for feature_fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for bagging_fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_freq in range(0, 50, 5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq
            cv_results = lgb.cv(params, lgb_train, seed=42, nfold=3, metrics=['mse'], early_stopping_rounds=3,verbose_eval=None)
            mean_merror = pd.Series(cv_results['l2-mean']).min()
            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

# params['feature_fraction'] = best_params['feature_fraction']
# params['bagging_fraction'] = best_params['bagging_fraction']
# params['bagging_freq'] = best_params['bagging_freq']
params = get_params(params, best_params)

# print("调参4：降低过拟合")
print('Params_4')
for lambda_l1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for lambda_l2 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            params['min_split_gain'] = min_split_gain
            cv_results = lgb.cv(params, lgb_train, seed=42, nfold=3, metrics=['mse'], early_stopping_rounds=3,verbose_eval=None)
            mean_merror = pd.Series(cv_results['l2-mean']).min()
            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
                best_params['min_split_gain'] = min_split_gain

# params['lambda_l1'] = best_params['lambda_l1']
# params['lambda_l2'] = best_params['lambda_l2']
# params['min_split_gain'] = best_params['min_split_gain']
params = get_params(params, best_params)
print('**********************************************************')
print(params)
print('**********************************************************')

params['learning_rate'] = 0.01

clf = lgb.train(params,  # 参数字典
    lgb_train,  # 训练集
    valid_sets=lgb_train,  # 验证集
    num_boost_round=6000,  # 迭代次数
    early_stopping_rounds=100, # 早停次数
    verbose_eval=None
    )
print('**********************************************************')
print(clf.feature_importance())
print('**********************************************************')

test = read_csv(path_test)
test_set = form_dataset(test)
df_test = test_set[feature].fillna(-1)
y_pred = clf.predict(df_test, num_iteration=clf.best_iteration)
result = pd.DataFrame(test_set['item'])
result['pre'] = y_pred
result = result.rename(columns={'item': 'Id', 'pre': 'Pred'})
result.to_csv('./model/result_.csv', header=True, index=False)
print('Predict Done!')

