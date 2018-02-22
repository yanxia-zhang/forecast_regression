# -*- coding: utf-8 -*-
# @Author: yanxia
# @Date:   2018-02-19 19:25:18
# @Last Modified by:   yanxia
# @Last Modified time: 2018-02-22 23:31:54


import pandas as pd
import numpy as np
import calendar, datetime


def read_and_preprocess_data(data_dir, file_name):
    dateparse = lambda dates: [pd.datetime.strptime(d, '%Y-%m-%d') for d in dates]
    df = pd.read_csv(data_dir+file_name, parse_dates = ['TSDate'], date_parser=dateparse)

    # Date time features
    df['year']      = df['TSDate'].apply(lambda x: x.year)
    df['quarter']   = df['TSDate'].apply(lambda x: x.quarter)
    df['month']     = df['TSDate'].apply(lambda x: x.month)
    df['week']      = df['TSDate'].apply(lambda x: x.week)
    df['weekday']   = df['TSDate'].apply(lambda x: x.weekday())

    # Convert seriesNames to int
    df['serieNames'] = df['serieNames'].str.extract('(\d+)', expand=False).astype(int)

    # Remove the TSDate columns
    df = df.set_index(['TSDate'])
    df = df.dropna()

    return df


def split_data(df, test_ratio = 0.1):
    train_size      = int(len(df) * (1.0 - test_ratio))
    train, test     = df[:train_size], df[train_size:]
    X_train, X_test = train.drop(['sales'], axis=1), test.drop(['sales'], axis=1)
    y_train, y_test = train['sales'], test['sales']

    return X_train, y_train, X_test, y_test


def prepare_seq_in_windows(seq, test_ratio, input_size, num_steps, normalized):
    # split into items of input_size
    seq = [np.array(seq[i * input_size: (i + 1) * input_size]) for i in range(len(seq) // input_size)]
    if normalized:
        seq = [seq[0] / seq[0][0] - 1.0] + [curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

    # split into groups of num_steps
    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])
    train_size = int(len(X) * (1.0 - test_ratio))

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test
