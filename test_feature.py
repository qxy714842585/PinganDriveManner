# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 18:31
# @Author  : qxy
# @File    : main_xgb.py.py
# @Software: PyCharm

import pandas as pd

from feature import form_dataset

path_train = '../PINGAN-2018-train_demo.csv'


feature = ['num_of_records', 'num_of_trips', 'num_of_state_0'
        , 'num_of_state_1', 'num_of_state_2','num_of_state_3', 'num_of_state_4'
        ,'mean_speed', 'var_speed', 'mean_height', 'var_height', 'tp0', 'tp1'
        , 'tp2', 'tp3', 'tp4', 'tp5', 'a0','a1', 'a2', 'a3', 'a4', 'a5', 'a6'
        , 'a7']
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


# def get_time(time_stamp):
#     time = datetime.datetime.fromtimestamp(time_stamp)
#     year = time.year
#     month = time.month
#     day = time.day
#     hour = time.hour
#     minute = time.minute
#     weekday = time.weekday()
#     return time
def train_model():
    train = read_csv(path_train)
    train_set = form_dataset(train)
    label = train_set['target']
    # params = {"objective": 'reg:linear', "eval_metric": 'rmse', "seed": 1123, "booster": "gbtree",
    #     "min_child_weight": 5, "gamma": 0.1, "max_depth": 3, "eta": 0.009, "silent": 1, "subsample": 0.65,
    #     "colsample_bytree": .35, "scale_pos_weight": 0.9# "nthread":16
    # }
    # df_train = xgb.DMatrix(train_set[feature].fillna(-1), label)
    # gbm = xgb.train(params, df_train, num_boost_round=1000)
    # pickle.dump(gbm, open("model.pickle.dat", "wb"))
    #todo use lightgbm regression model
    print("Model Trained!")

def process():
    train = read_csv(path_train)
    train_set = form_dataset(train)
    return train_set

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    # train = process()
