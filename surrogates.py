"""
Class to represent surrogate models for SAEA of Hyperparameter Optimization
Author: Mike Schladt 2023
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, ConstantKernel as C
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor

import numpy as np

class Surrogates():
    """
    SURROGATE MODELS
    """

    def __init__(self):
        """
        In our experiments, we will use a support vector regressor (svr) (with RBF kernel), 
        a gaussian process regressor (gpr) (with RBF + quadratic Kernels),
        and a gradient boosting regressor (gbr) as our surrogate models. 
        A voting ensemble will be used to combine the predictions of the three models. 
        Because we have two objectives, we will train two models for each surrogate model type.
        Models that begin with 1 are used to predict validation accuracy, 
        and models that begin with 2 are used to predict the loss target fitness.
        """

        self.svr1 = SVR(kernel='rbf', C=100, gamma="scale", epsilon=.1, degree=3)
        self.svr2 = SVR(kernel='rbf', C=100, gamma="scale", epsilon=.1, degree=3)
        kernel = C(1.0, (1e-3, 1e4)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(1e-2, 1e2)) + RBF(1.0, (1e-3, 1e4))
        self.gpr1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        self.gpr2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

        self.gbr1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42, loss="squared_error")
        self.gbr2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42, loss="squared_error")
        
        self.vr1 = VotingRegressor([('svr', self.svr1), ('gbr', self.gbr1), ('gpr', self.gpr1)])
        self.vr2 = VotingRegressor([('svr', self.svr2), ('gbr', self.gbr2), ('gpr', self.gpr2)])

    def train(self, X, y1, y2, verbose=True):
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
        train_features1, test_features1, train_labels1, test_labels1 = train_test_split(X, y1, test_size = 0.2, random_state = 42)

        # split the data into training and testing sets for loss target fitness
        train_features2, test_features2, train_labels2, test_labels2 = train_test_split(X, y2, test_size = 0.2, random_state = 42)

        # SVR validation accuracy
        self.svr1.fit(train_features1, train_labels1);
        predictions1 = self.svr1.predict(test_features1)
        self.svr1_mae = np.mean(abs(predictions1 - test_labels1))
        if verbose:
            print('SVR: Mean Absolute Error for validation accuracy:', round(self.svr1_mae, 2), '%.')

        # SVR loss target fitness
        self.svr2.fit(train_features2, train_labels2);
        predictions2 = self.svr2.predict(test_features2)
        self.svr2_mae = np.mean(abs(predictions2 - test_labels2))
        if verbose:
            print('SVR: Mean Absolute Error for loss target fitness:', round(self.svr2_mae, 2), '%.')

        # GBR validation accuracy
        self.gbr1.fit(train_features1, train_labels1);
        predictions1 = self.gbr1.predict(test_features1)
        self.gbr1_mae = np.mean(abs(predictions1 - test_labels1))
        if verbose:
            print('Gradient Boosting: Mean Absolute Error for validation accuracy:', round(self.gbr1_mae, 2), '%.')

        # GBR loss target fitness
        self.gbr2.fit(train_features2, train_labels2);
        predictions2 = self.gbr2.predict(test_features2)
        self.gbr2_mae = np.mean(abs(predictions2 - test_labels2))
        if verbose:
            print('Gradient Boosting: Mean Absolute Error loss target fitness:', round(self.gbr2_mae, 2), '%.')

        # GPR validation accuracy
        self.gpr1.fit(train_features1, train_labels1);
        predictions1 = self.gpr1.predict(test_features1)
        self.gpr1_mae1 = np.mean(abs(predictions1 - test_labels1))
        if verbose:
            print('Gaussian Process Regression: Mean Absolute Error for validation accuracy:', round(self.gpr1_mae1 , 2), '%.')

        # GP loss target fitness
        self.gpr2.fit(train_features2, train_labels2);
        predictions2 = self.gpr2.predict(test_features2)
        self.gpr1_mae2 = np.mean(abs(predictions2 - test_labels2))
        print('Gaussian Process Regression: Mean Absolute Error for loss target fitness:', round(self.gpr1_mae2, 2), '%.')

        # Voting Regressor validation accuracy
        self.vr1.fit(train_features1, train_labels1);
        predictions1 = self.vr1.predict(test_features1)
        self.vr1_mae = np.mean(abs(predictions1 - test_labels1))
        if verbose:
            print('Ensemble: Mean Absolute Error for validation accuracy:', round(self.vr1_mae, 2), '%.')

        # Voting Regressor loss target fitness
        self.vr2.fit(train_features2, train_labels2);
        predictions2 = self.vr2.predict(test_features2)
        self.vr2_mae = np.mean(abs(predictions2 - test_labels2))
        if verbose:
            print('Ensemble: Mean Absolute Error for loss target fitness:', round(self.vr2_mae, 2), '%.') 

    def predict(self, X):
        """
        Predicts the validation accuracy and loss target fitness for a given set of hyperparameters
        INPUT X: 3 dimensional numpy array (learning rate, momentum, weight decay)
        OUTPUT y1_pred: np array for predicted validation accuracies 
        OUTPUT y2_pred: np array for loss target fitnesses
        """
        # Standardize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Predict validation accuracy
        y1_pred = self.vr1.predict(X)

        # Predict loss target fitness
        y2_pred = self.vr2.predict(X)

        return y1_pred, y2_pred
    
    def get_mae(self):
        """
        Returns the mean absolute error for the validation accuracy and loss target fitness
        """
        mae_dict = {
            'svr1_mae': self.svr1_mae,
            'svr2_mae': self.svr2_mae,
            'gbr1_mae': self.gbr1_mae,
            'gbr2_mae': self.gbr2_mae,
            'gpr1_mae': self.gpr1_mae1,
            'gpr1_mae': self.gpr1_mae2,
            'vr1_mae': self.vr1_mae,
            'vr2_mae': self.vr2_mae
        }
        return mae_dict