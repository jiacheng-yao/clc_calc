import csv
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.ensemble
import pandas as pd
from collections import defaultdict
import numpy as np
import sklearn.preprocessing
import cPickle

from churn_prediction_comparison import load_data


class NaiveModel(object):
    """
    A class with the same interface as scikit-learn models but just returns the majority class
    label for all predictions
    """
    def __init__(self, majority_class):
        self.majority_class = majority_class

    def predict(self, x):
        return self.majority_class * np.ones(len(x))


def naive_model(X_train, X_test, y_train, y_test, features_columns):
    """
    Calculates the naive model accuracy
    :param X_train: None -- kept only to maintain a consistent interface
    :param X_test: None -- kept only to maintain a consistent interface
    :param y_train: None -- kept only to maintain a consistent interface
    :param y_test: List(Int) -- the test classes
    :param features_columns: None -- kept only to maintain a consistent interface
    :return: NaiveModel, float
    """

    # TODO Nico: The NaiveModel should be chosen using y_train data, and the accuracy should be
    # computed using the y_test data (since we are talking everywhere else about accuracy on the
    # test data set

    returning_users_ratio = float(y_train.sum()) / len(y_train)

    if returning_users_ratio > 0.5:
        cls = NaiveModel(1)
        accuracy = returning_users_ratio
    else:
        cls = NaiveModel(0)
        accuracy = 1 - returning_users_ratio
    print 'Proportion of users who return: {}'.format(returning_users_ratio)

    if returning_users_ratio == 0.0:
        print 'There are no returning users in the data. The training will fail'
    elif returning_users_ratio == 1.0:
        print 'There are only returning users in the data. The training will fail'

    return cls, accuracy


###########################################
# Logistic regression with added features
###########################################
def logistic_model(X_train, X_test, y_train, y_test, features_columns):
    """
    Trains a logistic regression model
    :param X_train: List(list)) -- the training features
    :param X_test: List(list)) -- the test features
    :param y_train: List -- the training targets
    :param y_test: List -- the test targets
    :param features_columns: None -- kept only to maintain a consistent interface
    :return: sklearn.linear_model.LogisticRegression, float
    """
    C_s = np.logspace(-3, 2, 10)

    max_accuracy = 0
    best_c = 0
    for C in C_s:
        cls = sklearn.linear_model.LogisticRegression(C=C)
        this_scores = sklearn.cross_validation.cross_val_score(
            cls, X_train, y_train, cv=2, n_jobs=1)
        score = np.mean(this_scores)

        if score > max_accuracy:
            best_c = C
            max_accuracy = score

    print 'Logistic regression (more features) best penalizer: {}'.format(best_c)
    cls = sklearn.linear_model.LogisticRegression(C=best_c)
    cls.fit(X_train, y_train)
    accuracy = cls.score(X_test, y_test)
    print 'Logistic regression (more features) accuracy: {}'.format(accuracy)

    return cls, accuracy


###########################################
# Gradient boosted classifier
###########################################
def gradient_boosted_classifier_model(X_train, X_test, y_train, y_test, features_columns):
    """
    Trains a gradient boosted classifier model
    :param X_train: List(list)) -- the training features
    :param X_test: List(list)) -- the test features
    :param y_train: List -- the training targets
    :param y_test: List -- the test targets
    :param features_columns: List(String) -- the names of the features
    :return: sklearn.ensemble.GradientBoostingClassifier, float
    """
    E_s = [60, 90, 150, 200]
    depths = [4, 5]

    max_accuracy = 0
    best_e = 0
    best_depth = 0

    for e in E_s:
        for d in depths:
            cls = sklearn.ensemble.GradientBoostingClassifier(n_estimators=e, max_depth=d)
            this_scores = sklearn.cross_validation.cross_val_score(
                cls, X_train, y_train, cv=2, n_jobs=1)
            score = np.mean(this_scores)

            if score > max_accuracy:
                best_e = e
                best_depth = d
                max_accuracy = score

    print 'Gradient boosting best estimators: {}'.format(best_e)
    print 'Gradient boosting best depth: {}'.format(best_depth)

    cls = sklearn.ensemble.GradientBoostingClassifier(n_estimators=best_e, max_depth=best_depth)
    cls.fit(X_train, y_train)
    accuracy = cls.score(X_test, y_test)
    print 'Gradient boosting accuracy: {}'.format(accuracy)

    print 'feature importances'
    for i in range(len(features_columns)):
        print '{0}: {1}'.format(features_columns[i], cls.feature_importances_[i])

    return cls, accuracy


###########################################
# Random Forest classifier
###########################################
def random_forest_model(X_train, X_test, y_train, y_test, features_columns):
    """
    Trains a random forest model
    :param X_train: List(list)) -- the training features
    :param X_test: List(list)) -- the test features
    :param y_train: List -- the training targets
    :param y_test: List -- the test targets
    :param features_columns: None -- kept only to maintain a consistent interface
    :return: sklearn.ensemble.RandomForestClassifier, float
    """
    E_s = [200]
    depths = [1]

    max_accuracy = 0
    best_e = 0
    best_depth = 0

    for e in E_s:
        for d in depths:
            cls = sklearn.ensemble.RandomForestClassifier(n_estimators=e)
            this_scores = sklearn.cross_validation.cross_val_score(
                cls, X_train, y_train, cv=2, n_jobs=1)
            score = np.mean(this_scores)

            if score > max_accuracy:
                best_e = e
                best_depth = d
                max_accuracy = score

    print 'Random forest best estimators: {}'.format(best_e)
    print 'Random forest best depth: {}'.format(best_depth)

    cls = sklearn.ensemble.RandomForestClassifier(n_estimators=best_e, max_depth=best_depth)
    cls.fit(X_train, y_train)
    accuracy = cls.score(X_test, y_test)
    print 'Random forest accuracy: {}'.format(accuracy)

    return cls, accuracy


def get_best_model(input_trans='/Users/yao.jiacheng/Documents/mix notebooks/dwh_il.fct_orders.csv',
                   input_custs='/Users/yao.jiacheng/Documents/mix notebooks/dwh_il.dim_customer.csv',
                   calibration_period_end='2016-04-30',
                   save_file_name='/Users/yao.jiacheng/Documents/mix notebooks/best_model_dh.pkl'):
    """
    Save & return the best model
    :param file_name: String -- the path of the training data
    :param save_file_name: String -- the path to save the model (as a cPickle file)
    :param scaler_save_file_name: String -- the path to save the scaler (as a cPickle file)
    :return: sklearn.model, sklearn.preprocessing.StandardScaler
    """
    X_train, X_test, y_train, y_test= load_data(input_trans, input_custs, calibration_period_end)

    features_columns = X_train.columns

    best_accuracy = 0
    i = 0
    best_i = 0

    model, accuracy = naive_model(X_train, X_test, y_train, y_test, features_columns)
    print 'Naive accuracy = {}'.format(accuracy)

    model_functions = [logistic_model, gradient_boosted_classifier_model, random_forest_model]

    for f in model_functions:
        model, accuracy = f(X_train, X_test, y_train, y_test, features_columns)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_i = i
        i += 1

    best_model, accuracy = model_functions[best_i](
        X_test, X_test, y_test, y_test, features_columns)

    with open(save_file_name, 'wb') as f:
        cPickle.dump(best_model, f)

    return best_model
