# -*- coding: utf-8 -*-
# @Author: yanxia
# @Date:   2018-02-18 20:33:00
# @Last Modified by:   yanxia
# @Last Modified time: 2018-02-23 00:40:04

import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
from sklearn.dummy import DummyRegressor

import dataset, models


def plot_prediction_perSerie(y_true, y_pred, y_serieNames):
    y = pd.DataFrame({  'Actual'    : y_true, 
                        'Pred'      : y_pred, 
                        'serieNames': y_serieNames
                    })
    y.groupby(['serieNames'])['Actual'].plot(lw = 2, label = 'Actual')
    y.groupby(['serieNames'])['Pred'].plot(style = 'o--', label = 'Pred')
    plt.legend(loc="best")


def Visualize_prediction(data_dir):
    # Visualize the prediction
    plt.figure()
    df_training         = pd.read_csv(data_dir+'training.csv')
    df_test_prediction  = pd.read_csv('./results/test_prediction.csv')
    final               = pd.concat([df_training, df_test_prediction], ignore_index=True)
    final               = final.set_index(['TSDate'])
    final.groupby(['serieNames'])['sales'].plot()
    plt.axvspan(0, len(df_training)*0.5*0.9, facecolor='k', alpha=0.1, label = 'train 90%')
    plt.axvspan(len(df_training)*0.5*0.9, len(df_training)*0.5, facecolor='y', alpha=0.1, label = 'test 10%')
    plt.axvspan(len(df_training)*0.5, len(final)*0.5, facecolor='g', alpha=0.3, label = 'prediction')
    plt.ylabel("sales")


def main():
    # ---------- Directories & User inputs --------------
    # Location of data folder
    data_dir    = './data/'
    FLAG_train  = (len(sys.argv) > 1 and sys.argv[1] == '--train')

    ##########################################
    ######## Load and preprocess data ########
    ##########################################

    # Read and preprocess data from CSV
    data = dataset.read_and_preprocess_data(data_dir = data_dir, file_name = 'training.csv')
    print data.head(), '\n', data.tail(), '\n', data.info()
    plt.figure()
    data.groupby(['serieNames'])['sales'].plot()
    plt.legend(loc="best")

    # Split data/labels into train/test set
    X_train, y_train, X_test, y_test        = dataset.split_data(df = data, test_ratio = 0.1)
    y_train_serieNames, y_test_serieNames   = X_train['serieNames'], X_test['serieNames']

    # Data normalization
    sc      = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)

    ##########################################
    ######## Train regressor #################
    ##########################################
    if FLAG_train:
        models.train_regressor_models(X_train, y_train, n_splits = 3)
    else:
        # Load the pre-trained regressor with tuned parameters
        # Linear Ridge Regression
        regressor_ridge = models.load_regressor_Ridge(X_train, y_train)
        # Random ForestRegression
        regressor_rf    = models.load_regressor_RF(X_train, y_train)
        # Support Vector Regression
        regressor_svr   = models.load_regressor_SVR(X_train, y_train)

        # Dummy regressor
        dummy       = DummyRegressor(strategy = 'mean')
        dummy.fit(X_train, y_train)
        y_hat_dummy = pd.DataFrame ({  
                                        'y_hat_dummy'   : y_test,
                                        'serieNames'    : y_test_serieNames
                                    })
        y_hat_dummy = y_hat_dummy.groupby(['serieNames'])['y_hat_dummy'].shift()
        y_hat_dummy = y_hat_dummy.fillna(method='bfill')
        print 'RMSE dummy mean %.5f' % (models.rmse(y_test, y_hat_dummy))

        ##########################################
        ######## Compare model performance #######
        ##########################################

        regressor_models =  {
                                'Baseline Previous Mean'    : dummy,
                                'Ridge Regression'          : regressor_ridge,
                                'Support Vector Regression' : regressor_svr,
                                'Random Forest Regression'  : regressor_rf
                            }

        # Test errors: test the model with tuned parameters
        for i, regressor_model in sorted(regressor_models.items()):
            y_hat_regressor = regressor_model.predict(X_test)
            RMSE_regressor  = models.rmse(y_test, y_hat_regressor)
            print 'RMSE %s : %.5f' % (i, RMSE_regressor)

            plt.figure()
            plt.ylabel("RMSE")
            plt.title('RMSE %s : %.5f' % (i, RMSE_regressor))
            plot_prediction_perSerie(y_true = y_test, y_pred = y_hat_regressor, y_serieNames = y_test_serieNames)

        plt.figure()
        plt.ylabel("RMSE")
        plt.title('RMSE dummy last observation %.5f' % (models.rmse(y_test, y_hat_dummy)))
        plot_prediction_perSerie(y_true = y_test, y_pred = y_hat_dummy, y_serieNames = y_test_serieNames)

        # Generization errors: cross_validate_score
        plt.figure()
        plt.title('Generalization errors (RMSE)')
        n_splits = 10
        scoring  = 'neg_mean_squared_error'
        for i, regressor_model in sorted(regressor_models.items()):
            test_error = models.get_regressor_cross_validate_score( regressor_model, 
                                                                    X_test, 
                                                                    y_test, 
                                                                    scoring = scoring, 
                                                                    n_splits = n_splits)
            test_rmse = np.array([np.sqrt(-e) for e in test_error])
            plt.plot(test_rmse, 'o-', label = i + ' : %0.2f (+/- %0.2f)' % (test_rmse.mean(), test_rmse.std() / 2))

        plt.xlabel("Fold number")
        plt.ylabel("RMSE")
        plt.legend(loc="best")

        ##########################################
        ######## Make predictions ################
        ##########################################
        file_name   = 'test.csv'
        new_samples = dataset.read_and_preprocess_data(data_dir = data_dir, file_name = file_name)

        X_new = new_samples.values
        X_new = sc.transform(X_new)

        # Directly predict
        y_new_hat = regressor_rf.predict(X_new)

        # Fit all data available and make prediction
        X_all = np.concatenate((X_train, X_test), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)
        regressor_rf.fit(X_all, y_all)
        y_new_hat_all = regressor_rf.predict(X_new)

        # Plot the prediction results
        plt.figure()
        df_new = pd.DataFrame({
                                'sales_pred_90' : y_new_hat,
                                'sales_pred_100': y_new_hat_all, 
                                'serieNames'    : new_samples['serieNames']
                              })
        df_new.groupby(['serieNames'])['sales_pred_90'].plot(label = 'sales_pred_90%')
        df_new.groupby(['serieNames'])['sales_pred_100'].plot(style = 'o--', label = 'sales_pred_100%')
        plt.ylabel("sales") 
        plt.legend(loc="best")

        ##########################################
        ######## Save prediction results #########
        ##########################################

        # Save the prediction results
        df_new.reset_index()
        df_new.to_csv('./results/prediction.csv', index=False)

        # Write to the test.csv format
        df_test = pd.read_csv(data_dir+'test.csv')
        df_test['sales'] = y_new_hat_all
        df_test.to_csv('./results/test_prediction.csv', index=False)

        plt.figure()
        df_test = df_test.set_index(['TSDate'])
        df_test.groupby(['serieNames'])['sales'].plot(style = '*-')
        plt.ylabel("sales")
        plt.legend(loc="best")

        # Visualize the prediction
        Visualize_prediction(data_dir)

        plt.legend(loc="best")
        plt.show()

if __name__ == "__main__":
    main()