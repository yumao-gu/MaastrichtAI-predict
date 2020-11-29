#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib
import numpy as np
import pandas  as pd
import seaborn as sns
import time

from datetime import date
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor

test_size = 0.15  # proportion of dataset to be used as test set
N = 3  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features

model_seed = 100


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def _load_data():
    stk_path = "/content/drive/MyDrive/Colab Notebooks/VTI.csv"
    df = pd.read_csv(stk_path, sep=",")
    # Convert Date column to datetime
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    # Get month of each sample
    df['month'] = df['date'].dt.month
    # Sort by datetime
    df.sort_values(by='date', inplace=True, ascending=True)

    return df


def feature_engineer(df):
    df['range_hl'] = df['high'] - df['low']
    df['range_oc'] = df['open'] - df['close']

    lag_cols = ['adj_close', 'range_hl', 'range_oc', 'volume', 'future_price']
    shift_range = [x + 1 for x in range(N)]

    for col in lag_cols:
        for i in shift_range:
            new_col = '{}_lag_{}'.format(col, i)  # 格式化字符串
            df[new_col] = df[col].shift(i)

    return df[N:]


def scale_row(row, feat_mean, feat_std):
    """
    Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
    Inputs
        row      : pandas series. Need to scale this.
        feat_mean: mean
        feat_std : standard deviation
    Outputs
        row_scaled : pandas series with same length as row, but scaled
    """
    # If feat_std = 0 (this happens if adj_close doesn't change over N days),
    # set it to a small number to avoid division by zero
    feat_std = 0.001 if feat_std == 0 else feat_std
    row_scaled = (row - feat_mean) / feat_std

    return row_scaled


def get_mov_avg_std(df, col, N):
    """
    Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe. Can be of any length.
        col        : name of the column you want to calculate mean and std dev
        N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
    Outputs
        df_out     : same as df but with additional column containing mean and std dev
    """
    mean_list = df[col].rolling(window=N, min_periods=1).mean()  # len(mean_list) = len(df)
    std_list = df[col].rolling(window=N, min_periods=1).std()  # first value will be NaN, because normalized by N-1

    # Add one timestep to the predictions ,这里又shift了一步
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

    # Append mean_list to df
    df_out = df.copy()
    df_out[col + '_mean'] = mean_list
    df_out[col + '_std'] = std_list

    return df_out


def get_future_price(df, col='adj_close', n=30):
    mean_list = df[col].rolling(window=n, min_periods=n).mean()  # len(mean_list) = len(df)
    # Append mean_list to df
    df_out = df.copy()
    df_out['future_price'] = mean_list
    df_out['future_price'] = df_out['future_price'].shift(-n)

    for i in range(len(df_out)):
        if df_out.loc[i, 'future_price'] > 0:
            if df_out.loc[i, col] - df_out.loc[i, 'future_price'] < -0.6:
                df_out.loc[i, 'class'] = 1
            elif df_out.loc[i, col] - df_out.loc[i, 'future_price'] > 0.6:
                df_out.loc[i, 'class'] = 2
            else:
                df_out.loc[i, 'class'] = 0
        else:
            df_out.loc[i, 'class'] = -1

    return df_out[:-n]


def get_predict_class(df):
    df_out = df.copy()
    for i in range(len(df_out)):
        # print(f'{df_out.iloc[i,:]}')
        if df_out.loc[df_out.index[i], 'pre_y1'] - df_out.loc[df_out.index[i], 'pre_y2'] < -0.6:
            df_out.loc[df_out.index[i], 'pre_class'] = 1
        elif df_out.loc[df_out.index[i], 'pre_y1'] - df_out.loc[df_out.index[i], 'pre_y2'] > 0.6:
            df_out.loc[df_out.index[i], 'pre_class'] = 2
        else:
            df_out.loc[df_out.index[i], 'pre_class'] = 0

    return df_out


if __name__ == '__main__':

    # 第一步：获取数据
    data_df = _load_data()

    # 第二步：特征工程
    data_df = get_future_price(data_df)
    df = feature_engineer(data_df)

    # 第三步：数据标准化，先统一计算出标准化的数据，在对其进行数据切分。
    cols_list = [
        "adj_close",
        "future_price",
        "range_hl",
        "range_oc",
        "volume"
    ]
    for col in cols_list:
        df = get_mov_avg_std(df, col, N)
    # print(f'{df.iloc[0,:]}')

    # 第四步：生成训练数据和测试数据。因训练数据和测试数据的标准化方式不同，因此需切分训练和测试数据。
    num_test = int(test_size * len(df))
    num_train = len(df) - num_test
    train = df[:num_train]
    test = df[num_train:]

    # 第五步：标签和特征的标准化，此步的目的是为了对在训练集不能代表总体的情况下，使树模型正确运行的一种取巧
    cols_to_scale = [
        "future_price",
        "adj_close"
    ]
    for i in range(1, N + 1):
        cols_to_scale.append("adj_close_lag_" + str(i))
        cols_to_scale.append("range_hl_lag_" + str(i))
        cols_to_scale.append("range_oc_lag_" + str(i))
        cols_to_scale.append("volume_lag_" + str(i))
        cols_to_scale.append("future_price_lag_" + str(i))

    scaler = StandardScaler()  # 启示三：标准化也不应带测试集，以避免信息泄漏
    train_scaled = scaler.fit_transform(train[cols_to_scale])
    # Convert the numpy array back into pandas dataframe
    train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
    train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]

    test_scaled = test[['date']]
    for col in tqdm(cols_list):
        feat_list = [col + '_lag_' + str(shift) for shift in range(1, N + 1)]
        temp = test.apply(lambda row: scale_row(row[feat_list], row[col + '_mean'], row[col + '_std']), axis=1)
        test_scaled = pd.concat([test_scaled, temp], axis=1)

    # 第六步：建立样本
    features = []
    for i in range(1, N + 1):
        features.append("adj_close_lag_" + str(i))
        features.append("range_hl_lag_" + str(i))
        features.append("range_oc_lag_" + str(i))
        features.append("volume_lag_" + str(i))
        features.append("future_price_lag_" + str(i))

    target1 = "adj_close"
    target2 = "future_price"

    X_train = train[features]
    y_train1 = train[target1]
    y_train2 = train[target2]
    X_sample = test[features]
    y_sample1 = test[target1]
    y_sample2 = test[target2]

    X_train_scaled = train_scaled[features]
    y_train_scaled1 = train_scaled[target1]
    y_train_scaled2 = train_scaled[target2]
    X_sample_scaled = test_scaled[features]

    # 第七步：开始训练
    from sklearn.model_selection import GridSearchCV

    parameters = {'n_estimators': range(50, 300, 50),
                  'max_depth': range(6, 11, 1),
                  'learning_rate': [0.2, 0.3, 0.4],
                  'min_child_weight': range(7, 12, 1),
                  # 'subsample':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  # 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  # 'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  # 'colsample_bylevel':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
                  }
    # parameters={'max_depth':range(2,10,1)}
    model1 = XGBRegressor(seed=model_seed,
                          n_estimators=100,
                          max_depth=3,
                          eval_metric='rmse',
                          learning_rate=0.1,
                          min_child_weight=1,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          gamma=0)
    gs1 = GridSearchCV(estimator=model1, param_grid=parameters, cv=5, refit=True, scoring='neg_mean_squared_error')
    gs1.fit(X_train_scaled, y_train_scaled1)

    model2 = XGBRegressor(seed=model_seed,
                          n_estimators=100,
                          max_depth=3,
                          eval_metric='rmse',
                          learning_rate=0.1,
                          min_child_weight=1,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          gamma=0)
    gs2 = GridSearchCV(estimator=model2, param_grid=parameters, cv=5, refit=True, scoring='neg_mean_squared_error')
    gs2.fit(X_train_scaled, y_train_scaled2)

    print('最优参数: ' + str(gs1.best_params_))
    print('最优参数: ' + str(gs2.best_params_))

    # est_scaled = gs.predict(X_train_scaled)
    # train['est'] = est_scaled * math.sqrt(scaler.var_[0]) + scaler.mean_[0]

    pre_y_scaled1 = gs1.predict(X_sample_scaled)
    test['pre_y_scaled1'] = pre_y_scaled1
    test['pre_y1'] = test['pre_y_scaled1'] * test['adj_close_std'] + test['adj_close_mean']

    plt.figure()
    ax = test.plot(x='date', y='adj_close', style='b-', grid=True)
    ax = test.plot(x='date', y='pre_y1', style='r-', grid=True, ax=ax)
    plt.show()

    pre_y_scaled2 = gs2.predict(X_sample_scaled)
    test['pre_y_scaled2'] = pre_y_scaled2
    test['pre_y2'] = test['pre_y_scaled2'] * test['future_price_std'] + test['future_price_mean']

    plt.figure()
    ax = test.plot(x='date', y='future_price', style='b-', grid=True)
    ax = test.plot(x='date', y='pre_y2', style='r-', grid=True, ax=ax)
    plt.show()

    test_new = get_predict_class(test)
    plt.figure()
    ax = test_new.plot(x='date', y='class', style='b-', grid=True)
    ax = test_new.plot(x='date', y='pre_class', style='r-', grid=True, ax=ax)
    plt.show()

    # rmse=math.sqrt(mean_squared_error(y_sample, test['pre_y']))
    # print("RMSE on dev set = %0.3f" % rmse)
    # mape = get_mape(y_sample, test['pre_y'])
    # print("MAPE on dev set = %0.3f%%" % mape)

    # imp = list(zip(train[features], gs.best_estimator_.feature_importances_))
    # imp.sort(key=lambda tup: tup[1])
    # for i in range(-1,-10,-1):
    #     print(imp[i])
