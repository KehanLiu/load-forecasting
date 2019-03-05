# -*- coding: utf-8 -*-
"""
author: hz@JN
updated: 2019/02/17

"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import config
import itertools


def dateparse(dates): 
    return pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')

def open_dataset_file(filename, folder = config.preprocessed_datasets_folder):
    
    data = pd.read_csv("prob_LSTM/" + folder + '/' + filename,
                       parse_dates=['timestamp'], index_col='timestamp', date_parser=dateparse)
    load = pd.DataFrame(
        data['watthour'], index=data.index, columns=['watthour'])
    # print(load)
    return(load)

def test_stationarity(timeseries):
    # 决定起伏统计
    rolmean = pd.rolling_mean(timeseries, window=12)    # 对size个数据进行移动平均
    rol_weighted_mean = pd.ewma(timeseries, span=12)    # 对size个数据进行加权移动平均
    rolstd = pd.rolling_std(timeseries, window=12)      # 偏离原始值多少
    # 画出起伏统计
    orig = plt.plot(timeseries, color = 'blue', label = 'Original')
    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    weighted_mean = plt.plot(
        rol_weighted_mean, color = 'green', label = 'weighted Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # 进行df测试
    print('Result of Dickry-Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print(dfoutput)


def ts_decomposing(ts_log):

    # 分解decomposing
    decomposition = seasonal_decompose(ts_log, freq=288)

    trend = decomposition.trend  # 趋势
    seasonal = decomposition.seasonal  # 季节性
    residual = decomposition.resid  # 剩余的

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonarity')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.tight_layout()

def ts_diff_describe(ts_log):
    # moving_avg = pd.rolling_mean(ts_log, 12)
    # ts_log_moving_avg_diff = ts_log - moving_avg
    # ts_log_moving_avg_diff.dropna(inplace=True)
    # test_stationarity(ts_log_moving_avg_diff)

    # ts_log_diff = ts_log.diff(1)
    # ts_log_diff.dropna(inplace = True)
    # test_stationarity(ts_log_diff)

    ts_log_diff1 = ts_log.diff(1)
    # 去除NA
    ts_log_diff1.dropna(inplace=True)
    ts_log_diff2 = ts_log.diff(2)
    ts_log_diff2.dropna(inplace=True)
    ts_log_diff1.plot()
    ts_log_diff2.plot()
    plt.show()


def regroup_time_preprocess(data):
    # 求和 小时计算
    y = data.resample('D').sum()
    
    #处理数据中的缺失项
    y = y.fillna(y.bfill())  # 填充缺失值
    plt.plot(y)
    plt.show()
    return(y)

def init_arima():
    # p是模型的自回归部分
    # d是模型的集成部分
    # q是模型的移动平均部分
    p = d = q = range(0, 2)
    print("p=", p, "d=", d, "q=", q)
    #产生不同的pdq元组,得到 p d q 全排列
    pdq = list(itertools.product(p, d, q))
    print("pdq:\n", pdq)
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    print('SQRIMAX:{} x {}'.format(pdq[1], seasonal_pdq[1]))
    return(pdq, seasonal_pdq)


def estimate_arima(pdq, seasonal_pdq, y):

    # pdq seasonal_pdq ARIMA中的参数
    # 网格搜索来迭代参数的不同组合

    best_pdq = pdq[0]
    best_s_pdq = seasonal_pdq[0]
    res_min = 10000

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

                if(results.aic < res_min):
                    res_min = results.aic
                    best_pdq = param
                    best_sq_pdq = param_seasonal
            except:
                continue
    
    print('Final ARIMA{}x{}12 - AIC:{}'.format(best_pdq, best_sq_pdq, res_min))

def fit_arima(y):
    """
    estimate_arima 获得的param, param_seasonal
    在这里输入
    """
    param = (1, 0, 1)
    param_seasonal = (0, 1, 1, 12)
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order = param,
                                    seasonal_order = param_seasonal,
                                    enforce_stationarity = False,
                                    enforce_invertibility = False)

    results = mod.fit()

    print(results.summary().tables[1])
    results.plot_diagnostics(figsize=(15, 12))
    # plt.show()
    return(results)

def pred_arima(pred_start_t, res, y):
    # @pred_start_t 动态预测时间标签
    # @res arima计算参数
    # @y 不分割的原始数据
    
    pred = res.get_prediction(
        start=pd.to_datetime(pred_start_t), dynamic=False)

    pred_ci = pred.conf_int()
    ax = y[pred_start_t:].plot(label="real")
    pred.predicted_mean.plot(ax=ax, label="static forcast",
                            alpha=.7, color='red', linewidth=5)
    #在某个范围内进行填充
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Load')
    plt.legend()
    plt.show()
    fig = plt.figure()
    plt.plot(y[pred_start_t:])
    plt.plot(pred.predicted_mean)
    y[pred_start_t:].to_csv("real.csv")
    pred.predicted_mean.to_csv("pred.csv")
    fig.savefig('output_arima_forecasting.png', bbox_inches='tight')


def main():
    data = open_dataset_file("DHW_3_granu300_6m_b.csv")
    preprocess_data = regroup_time_preprocess(data)
    # 求和 
    # pdq, seasonal_pdq = init_arima()
    # estimate_arima(pdq, seasonal_pdq, preprocess_data)
    res = fit_arima(preprocess_data)
    pred_arima('2017-10-15', res, preprocess_data)
    
    # estimating
    # ts_log = np.log(ts)

    # ts_diff_describe(ts_log)
    # ts_decomposing(ts_log)

if __name__ == "__main__":
    main()
