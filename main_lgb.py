# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 18:31
# @Author  : qxy
# @File    : main_xgb.py.py
# @Software: PyCharm


import lightgbm as lgb
import pandas as pd

from feature import form_dataset
from sklearn.model_selection import train_test_split

# path_train = '../PINGAN-2018-train_demo.csv'
path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件


feature = ['num_of_records', 'num_of_trips', 'num_of_state_0'
        , 'num_of_state_1', 'num_of_state_2','num_of_state_3', 'num_of_state_4'
        ,'mean_speed', 'var_speed', 'mean_height', 'var_height', 'tp0', 'tp1'
        , 'tp2', 'tp3', 'tp4', 'tp5', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'
        , 'a7', 'loc0', 'loc1', 'loc2', 'duration']
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


def train_model():
    train = read_csv(path_train)
    train_set = form_dataset(train)

    train, val = train_test_split(train_set, test_size=0.2, random_state=21)
    y_train = train['target']
    x_train = train[feature].fillna(-1)
    y_val = val['target']
    x_val = val[feature].fillna(-1)

    x = train_set[feature].fillna(-1)
    label = train_set['target']

    clf = lgb.LGBMRegressor(objective='regression', num_leaves=20, n_estimators=5000
                            , learning_rate=0.01, max_bin=55
                            # , random_state=2018
                            , bagging_fraction=0.8, bagging_freq=5
                            , feature_fraction=0.8, feature_fraction_seed=9, bagging_seed=9
                            )
    #todo boosting_type = 'dart'
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='mse', early_stopping_rounds=50)
    return clf
    # pickle.dump(clf, open("model.pickle.dat", "wb"))
    # print("Model Trained!")

def predict_y(model):
    test = read_csv(path_test)
    test_set = form_dataset(test)
    df_test = test_set[feature].fillna(-1)
    # loaded_model = pickle.load(open("model.pickle.dat", "rb"))
    loaded_model = model
    y_pred = loaded_model.predict(df_test)
    result = pd.DataFrame(test_set['item'])
    result['pre'] = y_pred
    result = result.rename(columns={'item': 'Id', 'pre': 'Pred'})
    result.to_csv('./model/result_.csv', header=True, index=False)
    print('Predict Done!')


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    model = train_model()
    predict_y(model)
    # train = process()
