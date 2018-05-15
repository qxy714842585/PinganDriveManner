# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 18:31
# @Author  : qxy
# @File    : main_xgb.py.py
# @Software: PyCharm


import pickle

import pandas as pd
import xgboost as xgb

from feature import form_dataset

# path_train = '../PINGAN-2018-train_demo.csv'
path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件


feature = ['num_of_records', 'num_of_trips', 'num_of_state_0'
    , 'num_of_state_1', 'num_of_state_2','num_of_state_3', 'num_of_state_4'
    ,'mean_speed', 'var_speed', 'mean_height', 'var_height'
    , 'tp0', 'tp1', 'tp2', 'tp3', 'tp4', 'tp5'
    , 'a0','a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'
    , 'loc0', 'loc1', 'loc2', 'duration'
    , 'turn_n0','turn_n1', 'turn_n2', 'turn_n3'
    , 'dis_pday']
# 34
# , 'night_drive'
#, 'sf0', 'sf1', 'sf2', 'sf3'
# , 'tp0', 'tp1', 'tp2', 'tp3', 'tp4', 'tp5'
# , 'loc_avg0', 'loc_avg1', 'loc_avg2'
# , 'steep0','steep1', 'steep2', 'steep3'
# , 'turn_n0','turn_n1', 'turn_n2', 'turn_n3'
#todo 4 add new feature name

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
    label = train_set['target']
    params = {"objective": 'reg:linear', "eval_metric": 'rmse', "seed": 1123, "booster": "gbtree",
        "min_child_weight": 5, "gamma": 0.1, "max_depth": 3, "eta": 0.01, "silent": 1, "subsample": 0.76,
        "colsample_bytree": .2, "scale_pos_weight": 0.9# "nthread":16
    }
    df_train = xgb.DMatrix(train_set[feature].fillna(-1), label)
    gbm = xgb.train(params, df_train, num_boost_round=1000)
    pickle.dump(gbm, open("model.pickle.dat", "wb"))
    print("Model Trained!")

def predict_y():
    test = read_csv(path_test)
    test_set = form_dataset(test)
    df_test = xgb.DMatrix(test_set[feature].fillna(-1))
    loaded_model = pickle.load(open("model.pickle.dat", "rb"))
    y_pred = loaded_model.predict(df_test)
    result = pd.DataFrame(test_set['item'])
    result['pre'] = y_pred
    result = result.rename(columns={'item': 'Id', 'pre': 'Pred'})
    result.to_csv('./model/result_.csv', header=True, index=False)
    print('Predict Done!')

# def process():
#     train = read_csv(path_train)
#     train_set = form_dataset(train)
#     return train_set

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    train_model()
    predict_y()
    # train = process()
