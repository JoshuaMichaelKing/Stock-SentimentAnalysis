#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from sklearn.linear_model import LinearRegression
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# import seaborn as sns

import iohelper
import stocktime as st

reload(sys)
sys.setdefaultencoding('utf-8')

# num of factors
num_of_closing_price_to_predicate = 5
# use emotion
# use_emotion = False
use_emotion = True
# forward unit of emotion
forward_unit = 1

# train date
date_of_march = ['20160329', '20160331']
date_of_april = ['20160405', '20160406', '20160407', '20160408',
'20160411', '20160412', '20160413', '20160414', '20160415',
'20160418', '20160419', '20160420', '20160421',
'20160425', '20160426', '20160427', '20160429']
date_of_may = ['20160503', '20160504', '20160505', '20160506',
'20160509', '20160510', '20160511', '20160512', '20160513',
'20160516', '20160517', '20160518', '20160519', '20160520',
'20160523', '20160524', '20160525', '20160526', '20160527',
'20160530', '20160531']
# test date
date_of_june = ['20160601', '20160602', '20160606', '20160613',
'20160614', '20160615', '20160620', '20160622', '20160624', '20160628']


def main():

    (train, test) = read_data()
    (arr_train_x, arr_train_y, arr_test_x, arr_test_y) = transform_original_data_2_train_data_and_test_data(train, test)
    global use_emotion
    global num_of_closing_price_to_predicate
    if not use_emotion:
        arr_train_x = arr_train_x[:, 0:num_of_closing_price_to_predicate]
        arr_test_x = arr_test_x[:, 0:num_of_closing_price_to_predicate]
    (fit_metrics, predict_metrics) = linear_regression(arr_train_x, arr_train_y, arr_test_x, arr_test_y)
    print_metrics(fit_metrics, predict_metrics)

def linear_regression(train_x, train_y, test_x, test_y):
    '''
    use linear regression to train and test data
    '''
    regressor = LinearRegression()
    regressor.fit(train_x, train_y)
    # 得到拟合的y值
    y_fit = np.dot(train_x, regressor.coef_ ) + regressor.intercept_
    # print('---'+str(regressor.coef_) + '---'+str(regressor.intercept_))
    # 计算拟合误差
    error_fit = regressor.score(train_x, train_y)
    rmse_fit = rmse(y_fit, train_y)
    mape_fit = mape(y_fit, train_y)

    # 预测
    y_predict = regressor.predict(test_x)
    error_predict = regressor.score(test_x, test_y)
    rmse_predict = rmse(y_predict, test_y)
    mape_predict = mape(y_predict, test_y)

    # 绘制对比图
    global num_of_closing_price_to_predicate
    dir_list = []
    dir_list.extend(date_of_march[1:])  # 跳过第一天
    dir_list.extend(date_of_april)
    dir_list.extend(date_of_may)
    day = 0
    begin = num_of_closing_price_to_predicate
    for one_day in dir_list:
        day_start = begin + day*48
        day_end = day_start + 48
        global forward_unit
        global use_emotion
        picture_name_suffix = ''
        if use_emotion:
            picture_name_suffix = str(forward_unit)+'emotion'
        picture(y_fit[day_start:day_end], train_y[day_start:day_end], 'train'+picture_name_suffix+one_day)
        if day_end <= len(y_predict):
            picture(y_predict[day_start:day_end], test_y[day_start:day_end], 'predict'+picture_name_suffix+date_of_june[day+1])
        day = day+1
    return (['error_fit_score: '+str(error_fit), 'rmse_fit_score: '+ str(rmse_fit), 'mape_fit_score: ' + str(mape_fit)], \
            ['error_predict_score: '+str(error_predict), 'rmse_predict_score: ' + str(rmse_predict), 'mape_predict_score: ' + str(mape_predict)])

###----  模型评价指标   ----
def rmse(ymodel, yreal):
    return sp.sqrt(sp.mean((ymodel - yreal) ** 2))

def mape(ymodel, yreal):
    return np.sum(np.true_divide(np.abs(ymodel-yreal), yreal))/len(ymodel)

###----   结果展示   ----
def print_metrics(fit_metrics, predict_metrics):
    for i in range(0, len(fit_metrics)):
        print('---' + fit_metrics[i] + '---' + predict_metrics[i] + '---')

def picture(y_model, y_real, name):
    '''
    绘制对比图
    '''
    # # 绘图初始化
    # fig_size = (12, 8)   # 图片尺寸
    # f, ax_array = plt.subplots(1, 2)  # 图片数量
    # f.set_size_inches(fig_size[0], fig_size[1])  # 设置图片尺寸

    # # 模型拟合y值对比图
    # sns.tsplot(data=y_fit, ax=ax_array[0])
    # sns.tsplot(data=train_y, ax=ax_array[0])

    # # 模型预测y值对比图
    # sns.tsplot(data=y_predict, ax=ax_array[1])
    # sns.tsplot(data=test_y, ax=ax_array[1])

    # plt.show()

    fig = plt.figure(figsize=(12,8))
    p1 = fig.add_subplot(111)
    # p1.xaxis.set_major_formatter(FuncFormatter(format_fn))
    # p1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    x = range(1, len(y_model)+1)
    p1.plot(x, y_model, 'o-', label="model", color="red", linewidth=3)
    p1.plot(x, y_real, 'o-', label="real", color="blue", linewidth=1)

    plt.title(name)
    plt.xlabel("time")
    plt.ylabel("closing price")
    plt.legend()
    filepath = './Pic/' + name + '.png'
    plt.savefig(filepath)

###----  数据读取  ----
def read_data():
    '''
    read closing data from files
    '''
    global date_of_march
    global date_of_april
    global date_of_may
    global date_of_june

    dir_list = []
    dir_list.extend(date_of_march)
    dir_list.extend(date_of_april)
    dir_list.extend(date_of_may)

    # 训练集
    original_train_data = [[], []]
    for subdir in dir_list:
        # 股指数据
        train_seq = iohelper.read_pickle2list(subdir, 'shindex_seq')
        original_train_data[0].extend(map(float, train_seq))
        # 情感数据
        train_seq = iohelper.read_pickle2list(subdir, 'saindex_seq')
        original_train_data[1].extend(map(float, train_seq))

    # 测试集
    original_test_data = [[], []]
    for subdir in date_of_june:
        # 股指数据
        test_seq = iohelper.read_pickle2list(subdir, 'shindex_seq')
        original_test_data[0].extend(map(float, test_seq))
        # 情感数据
        test_seq = iohelper.read_pickle2list(subdir, 'saindex_seq')
        original_test_data[1].extend(map(float, test_seq))

    return (original_train_data, original_test_data)

###----  数据转换  ----
def transform_original_data_2_train_data_and_test_data(original_train_data, original_test_data):
    '''
    transform original closing data to train data and test data:
    use multiple closing price to predict one closing price
    '''
    len_train_data = len(original_train_data[0])
    len_test_data = len(original_test_data[0])

    # 用于预测股指的收盘价数量
    global num_of_closing_price_to_predicate
    if len_train_data <= num_of_closing_price_to_predicate:
        print('ERROR: Closing original_train_data is not enough!')
        return 0

    # 生成训练和测试数据集
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    # 股指数据
    for i in range(0, num_of_closing_price_to_predicate):
        # 训练集
        stock_data = original_train_data[0][i:(len_train_data-num_of_closing_price_to_predicate+i)]
        train_x.append(stock_data)
        # 测试集
        stock_data = original_test_data[0][i:(len_test_data-num_of_closing_price_to_predicate+i)]
        test_x.append(stock_data)

    # 情感数据
      # 1. 训练集
    global forward_unit
    emotion_ratio = original_train_data[1][(num_of_closing_price_to_predicate-forward_unit):(len_train_data-forward_unit)]
    closing_data = original_train_data[0][(num_of_closing_price_to_predicate-1):(len_train_data-1)]
    emotion_data_train = map(lambda (a,b):a+b, zip(emotion_ratio,closing_data))
    train_x.append(emotion_data_train)
      # 2. 测试集
    emotion_ratio = original_test_data[1][(num_of_closing_price_to_predicate-forward_unit):(len_test_data-forward_unit)]
    closing_data = original_test_data[0][(num_of_closing_price_to_predicate-1):(len_test_data-1)]
    emotion_data_test = map(lambda (a,b):a+b, zip(emotion_ratio,closing_data))
    test_x.append(emotion_data_test)

    train_y = original_train_data[0][num_of_closing_price_to_predicate:len_train_data]
    test_y = original_test_data[0][num_of_closing_price_to_predicate:len_test_data]

    arr_train_x = (np.array(train_x)).T
    arr_train_y = np.array(train_y)
    arr_test_x = (np.array(test_x)).T
    arr_test_y = np.array(test_y)

    # 测试生成的数据集是否正确
    # print('------arr_train_x\'s shape ------' + str(arr_train_x.shape) + '------arr_train_x[0]\'s data ------' + str(arr_train_x[0]) )
    # print('------train_y\'s shape ------' + str(arr_train_y.shape) + '------arr_train_y[0]\'s data ------' + str(arr_train_y[0]) )
    # print('------arr_test_x\'s shape  ------' + str(arr_test_x.shape)  + '------arr_test_x[0]\'s data  ------' + str(arr_test_x[0]) )
    # print('------arr_test_y\'s shape  ------' + str(arr_test_y.shape)  + '------arr_test_y[0]\'s data  ------' + str(arr_test_y[0]) )

    return(arr_train_x, arr_train_y, arr_test_x, arr_test_y)

if __name__ == '__main__':
    main()
