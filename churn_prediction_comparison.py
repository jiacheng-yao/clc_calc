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
              input_custs='/Users/yao.jiacheng/Documents/mix notebooks/dwh_il.dim_customer.csv'):
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
                          'rider_tip_eur', 'promised_delivery_time', 'delivery_fee_vat_rate']

    for col in trans_data.columns:
        if trans_data.shape[0] - sum(trans_data[col].isnull().astype(int)) == 0:
            trans_cols_to_drop.append(col)

    trans_data.drop(trans_cols_to_drop, 1, inplace=True)

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

    data = pd.merge(trans_data, custs_data, how='inner', on=['customer_id'])