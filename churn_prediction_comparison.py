import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    confusion_matrix, r2_score, f1_score, roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data, customer_lifetime_value
from lifetimes import BetaGeoFitter, GammaGammaFitter, ModifiedBetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix, \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases, \
    plot_history_alive

from datetime import timedelta, date, datetime
from dateutil.rrule import rrule, MONTHLY
from plt_save import save


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def load_data(input_trans='/Users/yao.jiacheng/Documents/mix notebooks/dwh_il.fct_orders.csv',
              input_custs='/Users/yao.jiacheng/Documents/mix notebooks/dwh_il.dim_customer.csv',
              calibration_period_end='2016-05-31'):
    trans_data = pd.read_csv(input_trans, encoding='utf-8')
    custs_data = pd.read_csv(input_custs, encoding='utf-8')

    trans_cols_to_drop = ['rdbms_id', 'order_id', 'city_id', 'customer_ident',
                          'company_id', 'order_code_google', 'order_code_adjust',
                          'order_source', 'order_hour', 'status_id',
                          'decline_reason_id', 'vendor_id', 'chain_id', 'user_id',
                          'deliveryprovider_id', 'area_id', 'expected_delivery_time',
                          'paymenttype_id', 'online_payment', 'service_fee',
                          'delivery_fee', 'gmv_local', 'gfv_local', 'pc1_local',
                          'commission_local', 'fx', 'currency_code',
                          'voucher_used', 'voucher_id', 'voucher_value_local',
                          'voucher_value_eur', 'discount_used', 'discount_id',
                          'discount_value_local', 'discount_value_eur', 'first_order_all',
                          'first_order', 'first_app_order', 'preorder',
                          'status_date', 'surcharge_net_local', 'surcharge_net_eur',
                          'surcharge_gross_local', 'surcharge_gross_eur', 'rider_tip_local',
                          'delivery_address_id', 'edited', 'promised_expected_delivery_time',
                          'rider_tip_eur', 'promised_delivery_time', 'delivery_fee_vat_rate',
                          'expected_delivery_min']

    for col in trans_data.columns:
        if trans_data.shape[0] - sum(trans_data[col].isnull().astype(int)) == 0:
            trans_cols_to_drop.append(col)

    trans_data.drop(trans_cols_to_drop, 1, inplace=True)

    trans_data['minimum_delivery_value'] = trans_data['minimum_delivery_value'].fillna(0)

    custs_cols_to_drop = ['rdbms_id', 'customer_ident', 'customer_name',
                          'customer_email', 'email_domain', 'mobile_code',
                          'mobile_number', 'city', 'postcode',
                          'address_line1', 'address_line2', 'delivery_instructions',
                          'lat', 'lon', 'first_order_date_all', 'first_order_date',
                          'first_app_order_date', 'last_order_date', 'subscribe_date',
                          'code', 'company_name', 'updated_at',
                          'created_at', 'dwh_created', 'dwh_last_modified']
    for col in custs_data.columns:
        if custs_data.shape[0] - sum(custs_data[col].isnull().astype(int)) == 0:
            custs_cols_to_drop.append(col)

    custs_data.drop(custs_cols_to_drop, 1, inplace=True)

    custs_data['subscribed'] = custs_data['subscribed'].fillna(2)
    custs_data['source'] = custs_data['source'].apply(lambda x: x != 'b2c').astype(int)

    calibration_data = trans_data[trans_data['order_date'] <= calibration_period_end]

    holdout_data = trans_data[trans_data['order_date'] > calibration_period_end]

    other_cols = ['service_fee_eur', 'delivery_fee_eur', 'gmv_eur', 'gfv_eur',
                  'pc1_eur', 'commission_eur', 'minimum_delivery_value']

    calibration_summary = custom_summary_data_from_transaction_data(calibration_data, 'customer_id', 'order_date',
                                                                    other_cols,
                                                                    observation_period_end=calibration_period_end)

    holdout_summary = custom_summary_data_from_transaction_data(holdout_data, 'customer_id', 'order_date',
                                                                other_cols,
                                                                observation_period_end='2016-06-30')

    return calibration_summary, holdout_summary, custs_data


def split_data(calibration_summary, holdout_summary, custs_data):
    calibration_summary.reset_index(inplace=True)
    calibration_summary = pd.merge(calibration_summary, custs_data, how='inner', on=['customer_id'])
    calibration_summary.set_index('customer_id', inplace=True)

    real_customers_not_alive_index = list(
        set(calibration_summary['frequency'].index) - set(holdout_summary['frequency'].index))

    is_alive = pd.DataFrame(index=calibration_summary['frequency'].index)
    is_alive['real'] = 1

    is_alive.ix[real_customers_not_alive_index, 'real'] = 0

    X_train, X_test, y_train, y_test =\
        train_test_split(calibration_summary, is_alive['real'], test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test


def custom_find_first_transactions(transactions, customer_id_col, datetime_col, other_cols=None,
                                   datetime_format=None,
                                   observation_period_end=datetime.today(), freq='D'):
    """
    This takes a Dataframe of transaction data of the form:
        customer_id, datetime [, monetary_value]
    and appends a column named 'repeated' to the transaction log which indicates which rows
    are repeated transactions for that customer_id.
    Parameters:
        transactions: a Pandas DataFrame.
        customer_id_col: the column in transactions that denotes the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        other_cols: the columns in the transactions that denotes monetary value of the transaction.
            Optional, only needed for customer lifetime value estimation models.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    """
    select_columns = [customer_id_col, datetime_col]

    if other_cols:
        for col in other_cols:
            select_columns.append(col)

    transactions = transactions[select_columns].copy()

    # make sure the date column uses datetime objects, and use Pandas' DateTimeIndex.to_period()
    # to convert the column to a PeriodIndex which is useful for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col], format=datetime_format)
    transactions = transactions.set_index(datetime_col).to_period(freq)

    transactions = transactions.ix[(transactions.index <= observation_period_end)].reset_index()

    period_groupby = transactions.groupby([datetime_col, customer_id_col], sort=False, as_index=False)

    if other_cols:
        # when we have a monetary column, make sure to sum together any values in the same period
        period_transactions = period_groupby.sum()
    else:
        # by calling head() on the groupby object, the datetime_col and customer_id_col columns
        # will be reduced
        period_transactions = period_groupby.head(1)

    # initialize a new column where we will indicate which are the first transactions
    period_transactions['first'] = False
    # find all of the initial transactions and store as an index
    first_transactions = period_transactions.groupby(customer_id_col, sort=True, as_index=False).head(1).index
    # mark the initial transactions as True
    period_transactions.loc[first_transactions, 'first'] = True
    select_columns.append('first')

    return period_transactions[select_columns]


def custom_summary_data_from_transaction_data(transactions, customer_id_col, datetime_col, other_cols=None,
                                              datetime_format=None,
                                              observation_period_end=datetime.today(), freq='D'):
    """
    This transforms a Dataframe of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a Dataframe of the form:
        customer_id, frequency, recency, T [, monetary_value]
    Parameters:
        transactions: a Pandas DataFrame.
        customer_id_col: the column in transactions that denotes the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        other_cols: the columns in the transactions that denotes the monetary value of the transaction.
            Optional, only needed for customer lifetime value estimation models.
        observation_period_end: a string or datetime to denote the final date of the study. Events
            after this date are truncated.
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    """
    observation_period_end = pd.to_datetime(observation_period_end, format=datetime_format).to_period(freq)

    # label all of the repeated transactions
    repeated_transactions = custom_find_first_transactions(
        transactions,
        customer_id_col,
        datetime_col,
        other_cols,
        datetime_format,
        observation_period_end,
        freq
    )
    # count all orders by customer.
    customers = repeated_transactions.groupby(customer_id_col, sort=False)[datetime_col].agg(['min', 'max', 'count'])

    # subtract 1 from count, as we ignore their first order.
    customers['order_count'] = customers['count'] - 1
    customers['T'] = (observation_period_end - customers['min'])
    customers['frequency'] = customers['order_count'] / customers['T'].astype(float)
    customers['recency'] = (customers['max'] - customers['min'])

    summary_columns = ['order_count', 'recency', 'T', 'frequency']

    if other_cols:
        for col in other_cols:
            # create an index of all the first purchases
            first_purchases = repeated_transactions[repeated_transactions['first']].index
            # by setting the monetary_value cells of all the first purchases to NaN,
            # those values will be excluded from the mean value calculation
            repeated_transactions.loc[first_purchases, col] = np.nan

            customers[col] = repeated_transactions.groupby(customer_id_col)[col].mean().fillna(
                0)
            summary_columns.append(col)

    return customers[summary_columns].astype(float)
