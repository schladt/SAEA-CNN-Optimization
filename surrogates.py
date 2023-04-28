"""
Class to represent surrogate models for SAEA of Hyperparameter Optimization
Author: Mike Schladt 2023
"""

import numpy as np
from msvr.model.MSVR import MSVR # https://github.com/Analytics-for-Forecasting/msvr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import numpy as np

class Surrogates():
    """
    SURROGATE MODELS
    """

    def __init__(self):
        """
        In our experiments, we will use two multi target support vector regressors 
        (one with RBF kernel and one with laplacian kernel), we will also use a 
        random forest regressor. A voting ensemble will be used to combine the
        predictions of the three models.
        """

        self.msvr_rbf = MSVR(kernel = 'rbf', C=100)
        self.msvr_laplace = MSVR(kernel = 'laplacian', C=100)
        self.rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    def train(self, X, y, verbose=True):
        """
        Trains surrogate models
        INPUT X: training data features = 3 dimensional numpy array (learning rate, momentum, weight decay)
        INPUT y1: training data labels = 1 dimensional numpy array (validation accuracy)
        INPUT y2: training data labels = 1 dimensional numpy array (loss target fitness)
        INPUT verbose: boolean to print training results
        """
        # Standardize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # split the data into training and testing sets for validation accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # MSVR with RBF kernel
        self.msvr_rbf.fit(X_train, y_train)
        y_pred = self.msvr_rbf.predict(X_test)
        self.msvr_rbf_mae = np.mean(abs(y_pred - y_test))
        if verbose:
            print('MSVR (RBF): Mean Absolute Error:', round(self.msvr_rbf_mae, 2), '%.')

        # MSVR with Laplacian kernel
        self.msvr_laplace.fit(X_train, y_train)
        y_pred = self.msvr_laplace.predict(X_test)
        self.msvr_laplace_mae = np.mean(abs(y_pred - y_test))
        if verbose:
            print('MSVR (Laplacian): Mean Absolute Error:', round(self.msvr_laplace_mae, 2), '%.')

        # Random Forest Regressor
        self.rfr.fit(X_train, y_train)
        y_pred = self.rfr.predict(X_test)
        self.rfr_mae = np.mean(abs(y_pred - y_test))
        if verbose:
            print('Random Forest Regressor: Mean Absolute Error:', round(self.rfr_mae, 2), '%.')

    def predict(self, X):
        """
        Predicts the validation accuracy and loss target fitness for a given set of hyperparameters
        INPUT X: 3 dimensional numpy array (learning rate, momentum, weight decay)
        OUTPUT y_pred: 2 dimensional numpy array (validation accuracy, loss target fitness)
        """
        # Standardize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # make predictions with each model
        y_pred1 = self.msvr_rbf.predict(X)
        y_pred2 = self.msvr_laplace.predict(X)
        y_pred3 = self.rfr.predict(X)

        # get the average of each prediction
        y_pred = (y_pred1 + y_pred2 + y_pred3) / 3

        # limit the predictions to the range [0, 100]
        y_pred = np.clip(y_pred, 0, 100)

        return y_pred

    def get_mae(self):
        """
        Returns the mean absolute error for the validation accuracy and loss target fitness
        """
        mae_dict = {
            'msvr_rbf': self.msvr_rbf_mae,
            'msvr_laplace': self.msvr_laplace_mae,
            'rfr': self.rfr_mae
        }
        return mae_dict