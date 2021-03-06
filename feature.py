# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 13:37
# @Author  : qxy
# @File    : feature.py
# @Software: PyCharm

import datetime
from math import *

import pandas as pd


def get_user_num(data_set):
    user_num = data_set['TERMINALNO'].nunique()
    return user_num

def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def get_user_data(data, user_id):
    """
    输出特定用户的所有数据，并根据trip排序，再根据时间排序
    加入 distance 列数据，表示与上一次抽样点之间的距离 m 或 m/min
    :param data:
    :param user_id:
    :return:
    """
    # print('user NO:', user_id)
    user_data = data.loc[data['TERMINALNO'] == user_id, :]
    #temp 中存储了某个特定用户的所有数据
    user_data = user_data.sort_values(by=['TRIP_ID', 'TIME'])
    #按照TRIP_ID,TIME排序
    user_data.index = range(len(user_data))
    # todo wash the trip_id data
    trip_id = list(user_data['TRIP_ID'])
    lon = list(user_data['LONGITUDE'])
    lat = list(user_data['LATITUDE'])
    distance = [0]
    dis_sum = [0]
    time = list(user_data['TIME'])
    for i in range(1, len(lat)):
        if trip_id[i] == trip_id[i - 1]:
            if time[i]-time[i-1] != 0:
                distance.append(haversine1(lon[i], lat[i], lon[i - 1], lat[i - 1]) * 60 / (time[i] - time[i - 1]))
                dis_sum.append(haversine1(lon[i], lat[i], lon[i - 1], lat[i - 1]))
            else:
                distance.append(0)
                dis_sum.append(0)
        else:
            distance.append(0)
            dis_sum.append(0)
    # distance表示两次采样之间的距离，每个trip中的初始数据行 distance = 0
    # 也可以看作是单位 m/min 的速度
    user_data['DISTANCE'] = distance
    user_data['DIS_SUM'] = dis_sum
    return user_data


def generate_trip_ids(user_data):
    user_data = user_data.sort_values(by='TIME', ascending=True)
    last_time = -1
    trip_id = 0
    trip_ids = []
    for index, row in user_data.iterrows():
        if last_time == -1:
            last_time = row['TIME']
            trip_id += 1
            trip_ids.append(trip_id)
            continue
        if row['TIME'] - last_time > 600:
            trip_id += 1
        last_time = row['TIME']
        trip_ids.append(trip_id)
    user_data['TRIP_ID'] = trip_ids
    return user_data

# def get_user_data(data, user_id):
#     """
#     输出特定用户的所有数据，并根据trip排序，再根据时间排序
#     加入 distance 列数据，表示与上一次抽样点之间的距离 m 或 m/min
#     :param data:
#     :param user_id:
#     :return:
#     """
#     # print('user NO:', user_id)
#     user_data = data.loc[data['TERMINALNO'] == user_id, :]
#     #temp 中存储了某个特定用户的所有数据
#     user_data = generate_trip_ids(user_data)
#     trip_id = list(user_data['TRIP_ID'])
#     lon = list(user_data['LONGITUDE'])
#     lat = list(user_data['LATITUDE'])
#     time = list(user_data['TIME'])
#     distance = [0]
#     for i in range(1, len(lat)):
#         if trip_id[i] == trip_id[i - 1]:
#             if time[i]-time[i-1] != 0:
#                 distance.append(haversine1(lon[i], lat[i], lon[i - 1], lat[i - 1])*60/(time[i]-time[i-1]))
#             else:
#                 distance.append(0)
#         else:
#             distance.append(0)
#     # distance表示两次采样之间的距离，每个trip中的初始数据行 distance = 0
#     # 也可以看作是单位 m/min 的速度
#     user_data['DISTANCE'] = distance
#     return user_data

def get_record_num(user_data):
    record_num = user_data.shape[0]
    return [record_num]

def get_trip_num(user_data):
    trip_num = user_data['TRIP_ID'].nunique()
    return [trip_num]

def get_call_state(user_data):
    call_state = []
    call_state_time = []
    call_state_freq = []
    record_num = user_data.shape[0]
    for i in range(5):
        state_num = user_data.loc[user_data['CALLSTATE']==i].shape[0]
        call_state_time.append(state_num)
        state_freq = state_num/record_num
        call_state_freq.append(state_freq)
    call_state.extend(call_state_time)
    call_state.extend(call_state_freq)
    return call_state

def get_speed_feature(user_data):
    """
    利用speed特征计算平均速度和方差，采样速度不为 0 的值
    :param user_data:
    :return:
    """
    speed_feature = []
    mean_speed = user_data.loc[user_data['SPEED']>0]['SPEED'].mean()
    speed_feature.append(mean_speed)
    #var_speed = user_data.loc[user_data['SPEED']>0]['SPEED'].var()
    var_speed = user_data['SPEED'].var()
    speed_feature.append(var_speed)
    return speed_feature

def get_height_feature(user_data):
    height_feature = []
    mean_height = user_data['HEIGHT'].mean()
    height_feature.append(mean_height)
    var_height = user_data['HEIGHT'].var()
    height_feature.append(var_height)
    return height_feature

def get_time_period(user_data):
    record_num = user_data.shape[0]
    temp = pd.DataFrame()
    temp['hour'] = user_data['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    time_period_freq = []
    for i in range(6):
        #todo change time period
        t = i*4
        p_num = temp.loc[temp['hour'].isin([t, t+1, t+2, t+3])].shape[0]
        p_freq = p_num/record_num
        time_period_freq.append(p_freq)
    return time_period_freq

def get_location_area(user_data):
    record_num = user_data.shape[0]
    location_area = {'000':0, '001':0, '010':0, '011':0
        , '100':0, '101':0, '110':0, '111':0}
    area_freq = []
    x = list(user_data['LATITUDE'])
    y = list(user_data['LONGITUDE'])
    for i in range(record_num):
        if x[i] > 36:
            a = '0'
        else:
            a = '1'
        if y[i] < 95:
            b = '00'
        elif y[i] < 107:
            b = '01'
        elif y[i] < 118:
            b = '10'
        else:
            b = '11'
        area_key = a+b
        # if area_key not in location_area:
        #     location_area[area_key] = 0
        location_area[area_key] += 1
    for i in location_area:
        area_freq.append(location_area[i]/record_num)
    return area_freq

def get_duration(user_data):
    """
    平均每天出行时间  单位：秒
    :param user_data:
    :return:
    """
    temp = pd.DataFrame()
    temp['date'],temp['time'],temp['TRIP_ID'] = user_data['TIME'].apply(lambda x:datetime.date.fromtimestamp(x)),user_data['TIME'],user_data['TRIP_ID']
    grouped = temp['time'].groupby([temp['date'],temp['TRIP_ID']])
    a = grouped.max() - grouped.min()
    duration = a.sum(level = 'date').mean()
    return [duration]


def get_speed_freq(user_data):
    """
    停低中高时间占比
    :param user_data:
    :return:    返回有4个元素的列表，4元素之和为1
    """
    trip_num = user_data['TRIP_ID'].nunique()
    record_num = user_data.shape[0]-trip_num
    a = lambda x:user_data[user_data['DISTANCE'] < x*1000/60]['TIME'].count()
    speed_times = [a(1)-trip_num, a(30)-a(1), a(60)-a(30), record_num-a(60)+trip_num]
    speed_freq = list(map(lambda x: x/record_num, speed_times))
    return speed_freq


def get_location(user_data):
    freq = get_location_area(user_data)
    freq_dict = {'000': 0, '001': 0, '010': 0, '011': 0
        , '100': 0, '101': 0, '110': 0, '111': 0}
    for i in freq_dict:
        freq_dict[i] = freq.pop(0)
    a = sorted(freq_dict.items(), key=lambda e:e[1])
    b = a[0][0]
    return [int(i) for i in b]

def get_location_avg(user_data):
    freq = get_location_area(user_data)
    freq_dict = {'000': 0, '001': 0, '010': 0, '011': 0
        , '100': 0, '101': 0, '110': 0, '111': 0}
    for i in freq_dict:
        freq_dict[i] = freq.pop(0)
    a = sorted(freq_dict.items(), key=lambda e:e[1])
    b = a[0][0]
    return [int(i) for i in b]


def get_time_period_24(user_data):
    record_num = user_data.shape[0]
    temp = pd.DataFrame()
    temp['hour'] = user_data['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    time_period_freq = []
    for i in range(24):
        p_num = temp.loc[temp['hour']==i].shape[0]
        p_freq = p_num/record_num
        time_period_freq.append(p_freq)
    return time_period_freq


def get_steep_duration(user_data):
    """
    计算陡坡    陡坡判定 上：>tan.max()/4 下： <tan.min()/4
    :param user_data:
    :return: steep_duration = [total_up, avg_up, total_down, avg_down]
    """
    trip_id = list(user_data['TRIP_ID'])
    h = list(user_data['HEIGHT'])
    t = list(user_data['TIME'])
    dH, dt, tan = [0], [0], [0]

    for i in range(1, len(h)):
        if trip_id[i] == trip_id[i - 1]:
            dH.append(h[i] - h[i - 1])
            dt.append(t[i] - t[i - 1])
        else:
            dH.append(0)
            dt.append(0)
        if user_data['DISTANCE'][i] == 0:
            tan.append(0)
        else:
            tan.append(dH[i] / user_data['DIS_SUM'][i])

    temp = pd.DataFrame()
    temp['trip_id'] = user_data['TRIP_ID']
    temp['dH'] = dH
    temp['tan'] = tan
    temp['dt'] = dt

    total_up = temp[temp['tan'] > 0.2]['dt'].sum()
    avg_up = total_up / user_data['TRIP_ID'].nunique()

    total_down = temp[temp['tan'] < -0.2]['dt'].sum()
    avg_down = total_down / user_data['TRIP_ID'].nunique()
    steep_duration = [total_up, avg_up, total_down, avg_down]
    return steep_duration

# ********************************转弯次数**************************************

def get_azimuth(lon0, lat0, lon1, lat1):
    distance_North = lat1 - lat0
    distance_East = -cos(lat0) * (lon1 - lon0)
    bearing = atan2(distance_East, distance_North)
    return bearing

def get_angle_diff(angle2, angle1):
    diff = angle2 - angle1
    while diff > pi:
        diff -= 2 * pi
    while diff < -pi:
        diff += 2 * pi
    return diff

def get_turn_num(trip_data):
    last_lon = -1
    last_lat = -1
    direction = []
    for index, row in trip_data.iterrows():
        if last_lon == -1 and last_lat == -1:
            last_lon = row['LONGITUDE']
            last_lat = row['LATITUDE']
            continue
        lon = row['LONGITUDE']
        lat = row['LATITUDE']
        if haversine1(last_lon, last_lat, lon, lat) > 50:
            direction.append(get_azimuth(last_lon, last_lat, lon, lat))
        last_lon = lon
        last_lat = lat
    turn_left = 0
    turn_right = 0
    turn_round = 0
    jump_next = False
    for i in range(len(direction) - 1):
        if jump_next:
            jump_next = False
            continue
        diff = get_angle_diff(direction[i + 1], direction[i])
        if abs(diff) > 150 * pi / 180:
            turn_round += 1
        elif diff > 35 * pi / 180:
            turn_right += 1
            jump_next = True
        elif diff < -35 * pi / 180:
            turn_left += 1
            jump_next = True
    return [turn_left, turn_right, turn_left + turn_right, turn_round]

# return [turn_left, turn_right, turn_sum, turn_round]
def get_avg_turn_num(user_data):
    trip_list = user_data['TRIP_ID'].unique()
    turn = [0, 0, 0, 0]
    for trip in trip_list:
        ret = get_turn_num(user_data.loc[user_data['TRIP_ID'] == trip, :].sort_values(by='TIME', ascending=True))
        for i in range(4):
            turn[i] += ret[i]
    for i in range(4):
        turn[i] = turn[i] * 1.0 / len(trip_list)
    return turn

# ********************************************************************

def get_Height_change(user_data):
    trip_id = list(user_data['TRIP_ID'])
    h = list(user_data['HEIGHT'])
    dH = [0]
    for i in range(1, len(h)):
        if trip_id[i] == trip_id[i - 1]:
            dH.append(h[i] - h[i-1])
        else:
            dH.append(0)
    temp = pd.DataFrame()
    temp['dH'] =  dH
    temp['trip_id'] = trip_id
    up_dH = sum(temp[temp['dH'] > 0]['dH'])/temp['trip_id'].nunique()
    down_dH = -sum(temp[temp['dH'] < 0]['dH'])/temp['trip_id'].nunique()
    total_dH = up_dH + down_dH
    Height_change = [up_dH, down_dH, total_dH]
    return Height_change


def get_night_drive(user_data):
    """
    #日均夜车条数(23:00-5:00)
	:param user_data:
	:return:
	"""
    temp = pd.DataFrame()
    temp['hour'] = user_data['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    temp['date'] = user_data['TIME'].apply(lambda x:datetime.date.fromtimestamp(x))
    num_of_night_drive = temp.loc[temp['hour'].isin([23, 0, 1, 2, 3, 4])].shape[0]
    night_drive = num_of_night_drive/temp['date'].nunique()
    return [night_drive]


def get_dis_per_day(user_data):
    """
    user每日里程
	:param user_data:
	:return:
	"""
    temp = pd.DataFrame()
    temp['date'] = user_data['TIME'].apply(lambda x:datetime.date.fromtimestamp(x))
    dis_per_day = user_data['DIS_SUM'].sum()/temp['date'].nunique()
    return [dis_per_day]

def get_dH_per_day(user_data):
    trip_id = list(user_data['TRIP_ID'])
    h = list(user_data['HEIGHT'])
    dH = [0]
    for i in range(1, len(h)):
        if trip_id[i] == trip_id[i - 1]:
            dH.append(h[i] - h[i-1])
        else:
            dH.append(0)
    temp = pd.DataFrame()
    temp['dH'] =  dH
    temp['date'] = user_data['TIME'].apply(lambda x:datetime.date.fromtimestamp(x))
    up_dH = sum(temp[temp['dH'] > 0]['dH'])/temp['date'].nunique()
    down_dH = -sum(temp[temp['dH'] < 0]['dH'])/temp['date'].nunique()
    total_dH = up_dH + down_dH
    dH_per_day = [up_dH, down_dH, total_dH]
    return dH_per_day


#todo 1 add new feature function before this line

def get_target(user_data):
    try:
        target = user_data['Y'].mean()
    except:
        target = -1
    return [target]


def form_user_feature(user_data, user_id):
    user_feature = []
    user_feature.append(user_id)
    user_feature.extend(get_record_num(user_data))
    user_feature.extend(get_trip_num(user_data))
    user_feature.extend(get_call_state(user_data))
    user_feature.extend(get_speed_feature(user_data))
    user_feature.extend(get_height_feature(user_data))
    user_feature.extend(get_time_period(user_data))
    user_feature.extend(get_location_area(user_data))

    user_feature.extend(get_duration(user_data))
    user_feature.extend(get_speed_freq(user_data))
    user_feature.extend(get_location(user_data))
    user_feature.extend(get_time_period_24(user_data))
    user_feature.extend(get_location_avg(user_data))
    user_feature.extend(get_steep_duration(user_data))
    user_feature.extend(get_avg_turn_num(user_data))
    user_feature.extend(get_Height_change(user_data))
    user_feature.extend(get_night_drive(user_data))
    user_feature.extend(get_dis_per_day(user_data))
    user_feature.extend(get_dH_per_day(user_data))
        #todo 2 add new feature list before this line

    user_feature.extend(get_target(user_data))
    return user_feature

def form_dataset(data):
    print(get_user_num(data))
    user_list = data['TERMINALNO'].unique()
    data_list = []
    for user_id in user_list:
        user_data = get_user_data(data, user_id)
        data_list.append(form_user_feature(user_data, user_id))
    data_set = pd.DataFrame(data_list)
    feature_name = ['item', 'num_of_records', 'num_of_trips','time_of_state_0'
        , 'time_of_state_1', 'time_of_state_2','time_of_state_3', 'time_of_state_4'
        , 'num_of_state_0'
        , 'num_of_state_1', 'num_of_state_2','num_of_state_3', 'num_of_state_4'
        ,'mean_speed', 'var_speed', 'mean_height', 'var_height', 'tp0', 'tp1'
        , 'tp2', 'tp3', 'tp4', 'tp5', 'a0','a1', 'a2', 'a3', 'a4', 'a5', 'a6'
        , 'a7', 'duration', 'sf0', 'sf1', 'sf2', 'sf3', 'loc0', 'loc1', 'loc2'
        , '24tp0', '24tp1', '24tp2', '24tp3', '24tp4', '24tp5', '24tp6', '24tp7'
        , '24tp8', '24tp9', '24tp10', '24tp11', '24tp12', '24tp13', '24tp14', '24tp15'
        , '24tp16', '24tp17', '24tp18', '24tp19', '24tp20', '24tp21', '24tp22'
        , '24tp23', 'loc_avg0', 'loc_avg1', 'loc_avg2', 'steep0','steep1', 'steep2'
        , 'steep3', 'turn_n0','turn_n1', 'turn_n2', 'turn_n3', 'H_up', 'H_down', 'H_total'
        , 'night_drive', 'dis_pday', 'up_dH', 'down_dH', 'total_dH', 'target']
    #todo 3 add new feature name before 'target'
    try:
        data_set.columns = feature_name
    except:
        data_set.columns = feature_name[:-1]
    print ('DataSet Done!')
    return data_set
