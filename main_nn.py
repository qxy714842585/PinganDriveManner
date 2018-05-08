# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 10:15
# @Author  : qxy
# @File    : main_nn.py
# @Software: PyCharm

import numpy
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

from feature import form_dataset

# path_train = '../PINGAN-2018-train_demo.csv'
path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件


feature = ['num_of_records', 'num_of_trips', 'num_of_state_0'
        , 'num_of_state_1', 'num_of_state_2','num_of_state_3', 'num_of_state_4'
        ,'mean_speed', 'var_speed', 'mean_height', 'var_height', 'tp0', 'tp1'
        , 'tp2', 'tp3', 'tp4', 'tp5', 'a0','a1', 'a2', 'a3', 'a4', 'a5', 'a6'
        , 'a7', 'duration', 'sf0', 'sf1', 'sf2', 'sf3']
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



def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=25, init='normal', activation='relu'))
    #todo 25 improve to 30,40 etc.
    #todo add dropout layer
    # model.add(Dropout(0.2))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model():
    train = read_csv(path_train)
    train_set = form_dataset(train)
    label = train_set['target']
    #数据标准化
    scaler = StandardScaler().fit(train_set[feature].fillna(-1))
    x = scaler.transform(train_set[feature].fillna(-1))
    x = pd.DataFrame(x)
    seed = 7
    # fix random seed for reproducibility
    numpy.random.seed(seed)
    model = baseline_model()
    model.fit(x, label, batch_size=1000, epochs=200, validation_split=0.2)
    print("Model Trained!")
    return scaler, model

def predict_y(scaler, model):
    test = read_csv(path_test)
    test_set = form_dataset(test)
    x = scaler.transform(test_set[feature].fillna(-1))
    x = pd.DataFrame(x)
    y_pred = model.predict(x)
    result = pd.DataFrame(test_set['item'])
    result['pre'] = y_pred
    result = result.rename(columns={'item': 'Id', 'pre': 'Pred'})
    result.to_csv('./model/result_.csv', header=True, index=False)
    print('Predict Done!')

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    # train = process()
    scaler, model = train_model()
    predict_y(scaler, model)