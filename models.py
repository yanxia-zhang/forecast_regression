# -*- coding: utf-8 -*-
# @Author: yanxia
# @Date:   2018-02-19 19:26:02
# @Last Modified by:   yanxia
# @Last Modified time: 2018-02-23 00:03:55

import pandas as pd
import numpy as np
import pickle
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error#, mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# from sklearn.dummy import DummyRegressor


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_regressor_Ridge(X_train, y_train, scoring, n_splits = None):
    if n_splits == None:
        model = Ridge()
        model.fit(X_train, y_train)
    elif n_splits > 0:
        # Split the time series data into training and validation sets
        ts_cv  = TimeSeriesSplit(n_splits).split(X_train)
        # Tuning the parameters
        alphas = np.array([100,10,1,0,0.1,0.01,0.001])
        model  = GridSearchCV(  Ridge(), 
                                param_grid  = dict(alpha=alphas), 
                                scoring     = scoring, 
                                cv          = ts_cv, 
                                n_jobs      = -1
                             )
        model.fit(X_train, y_train)
        print model.best_params_

    # Save it to a file
    joblib.dump(model.best_params_, './built_models/regressor_ridge.pkl')

    return model


def load_regressor_Ridge(X_train, y_train):
    best_estimator_ridge = joblib.load('./built_models/regressor_ridge.pkl')    
    regressor_ridge      = Ridge()
    regressor_ridge.set_params(**best_estimator_ridge)
    print regressor_ridge
    regressor_ridge.fit(X_train, y_train)

    return regressor_ridge


def build_regressor_RF(X_train, y_train, scoring, n_splits = None):
    if n_splits == None:
        model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
        model.fit(X_train, y_train)
    elif n_splits > 0:
        # Split the time series data into training and validation sets
        ts_cv = TimeSeriesSplit(n_splits).split(X_train)
        # Tuning the parameters 
        param_grid = {
                        "n_estimators"  : [200, 500, 1000],
                        "max_depth"     : [3, None],
                        "max_features"  : ['auto', 'sqrt', 'log2', None],
                        "bootstrap"     : [True, False]
                     }
        model = GridSearchCV(   estimator   = RandomForestRegressor(random_state=0), 
                                param_grid  = param_grid, 
                                scoring     = scoring, 
                                cv          = ts_cv, 
                                n_jobs      = -1
                            )
        model.fit(X_train, y_train)
        print model.best_params_

    # Save it to a file
    joblib.dump(model.best_params_, './built_models/regressor_RF.pkl')

    return model


def load_regressor_RF(X_train, y_train):
    best_estimator_rf   = joblib.load('./built_models/regressor_RF.pkl')  
    regressor_rf        = RandomForestRegressor()
    regressor_rf.set_params(**best_estimator_rf)
    print regressor_rf
    regressor_rf.fit(X_train, y_train)

    return regressor_rf


def build_regressor_SVR(X_train, y_train, scoring, n_splits = None):
    if n_splits == None:
        model = SVR(kernel='linear', C=1e3)
        model.fit(X_train, y_train)
    elif n_splits > 0:
        # Split the time series data into training and validation sets
        ts_cv = TimeSeriesSplit(n_splits).split(X_train)
        # Tuning the parameters 
        param_grid = {
                        "C"     : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        "gamma" : [0.001, 0.01, 0.1, 1]
                     }
        model = GridSearchCV(   SVR( kernel = 'rbf'), 
                                param_grid  = param_grid, 
                                scoring     = scoring, 
                                cv          = ts_cv, 
                                n_jobs      = -1
                            )
        model.fit(X_train, y_train)
        print model.best_params_

    # Save it to a file
    joblib.dump(model.best_params_, './built_models/regressor_SVR.pkl')

    return model


def load_regressor_SVR(X_train, y_train):
    best_estimator_svr  = joblib.load('./built_models/regressor_SVR.pkl')    
    regressor_svr       = SVR()
    regressor_svr.set_params(**best_estimator_svr)
    print regressor_svr
    regressor_svr.fit(X_train, y_train)

    return regressor_svr


def get_regressor_cross_validate_score(model, X_test, y_test, scoring, n_splits):
    ts_cv = TimeSeriesSplit(n_splits).split(X_test)
    score = cross_validation.cross_val_score(model, X_test, y_test, cv=ts_cv, scoring=scoring, n_jobs=-1)
    print 'Best %s: %0.2f (+/- %0.2f)' % (scoring, score.mean(), score.std() / 2)

    return score


def train_regressor_models(X_train, y_train, n_splits, scoring = 'neg_mean_squared_error'):
    # Linear ridge regression
    regressor_ridge = build_regressor_Ridge( X_train  = X_train, 
                                             y_train  = y_train, 
                                             scoring  = scoring, 
                                             n_splits = n_splits
                                           )
    # Random forest regression
    regressor_rf    = build_regressor_RF(    X_train  = X_train, 
                                             y_train  = y_train, 
                                             scoring  = scoring, 
                                             n_splits = n_splits
                                        )
    # Support vector regression
    regressor_svr   = build_regressor_SVR(   X_train  = X_train, 
                                             y_train  = y_train, 
                                             scoring  = scoring, 
                                             n_splits = n_splits
                                         )


def predict_regressor(X_test, model):
    y_hat = model.predict(X_test)
    error = models.rmse(y_test, y_hat)
    print 'Error %.5f' % (error)
    print np.sqrt(-model.score(X_test, y_test))

    return y_hat