import pandas as pd

from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix, \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases, \
    plot_history_alive

import multiprocessing

from datetime import timedelta, date
from plt_save import save


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

is_summary_available = True
input_file = "FD DE customer orders and revenue (1).xlsx"
output_file = "summary_fd_de_customers.csv"

plot_source = "fd"

if is_summary_available is False:
    # transaction_data = pd.read_csv(input_file, sep=';')
    transaction_data = pd.read_excel(input_file, sep=';') # for FD dataset

    recent_transaction_data = transaction_data[transaction_data['order_date'] > "2015-07-31"]

    summary = summary_data_from_transaction_data(recent_transaction_data,
                                                 'customer_id', 'order_date',
                                                 'revenue', observation_period_end='2016-08-03')

    print summary.head()

    summary.to_csv(output_file, sep=';', encoding='utf-8')
else:
    transaction_data = pd.read_csv(input_file, sep=';')

    recent_transaction_data = transaction_data[transaction_data['order_date'] > "2015-07-31"]

    summary = pd.read_csv(output_file, sep=';')


def bgf_analyser(summary, plot_source):
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary['frequency'],
            summary['recency'],
            summary['T'])
    print bgf

    plot_frequency_recency_matrix(bgf)
    save("f_r_matrix_{}".format(plot_source), ext="pdf", close=True, verbose=True)

    plot_probability_alive_matrix(bgf)
    save("pr_alive_matrix_{}".format(plot_source), ext="pdf", close=True, verbose=True)

    plot_period_transactions(bgf)
    save("period_transactions_{}".format(plot_source), ext="pdf", close=True, verbose=True)


def bgf_history_alive(transaction_data, bgf, customer_id='80_3811'):
    id = customer_id
    days_since_birth = 365
    sp_trans = transaction_data.ix[transaction_data['customer_id'] == id]
    plot_history_alive(bgf, days_since_birth, sp_trans, 'order_date')
    save("history_alive_{}".format(plot_source), ext="pdf", close=True, verbose=True)


def cal_vs_holdout_in_parallel(transaction_data, calibration_period_end='2016-05-01'):
    bgf_ = BetaGeoFitter(penalizer_coef=0.0)
    summary_cal_holdout = calibration_and_holdout_data(transaction_data, 'customer_id', 'order_date',
                                                           calibration_period_end=calibration_period_end,
                                                           observation_period_end='2016-08-03')
    print summary_cal_holdout.head()
    bgf_.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])

    plot_calibration_purchases_vs_holdout_purchases(bgf_, summary_cal_holdout)
    save("cal_vs_holdout_{}".format(calibration_period_end), ext="pdf", close=True, verbose=True)

start_date = date(2015, 10, 1)
end_date = date(2016, 4, 1)

periods = [single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)]

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
results = [pool.apply_async(func=cal_vs_holdout_in_parallel, args=(transaction_data, period)).get() for period in periods]


def ggf_analyser(summary, bgf):
    returning_customers_summary = summary[summary['frequency'] > 0]

    print returning_customers_summary[['monetary_value', 'frequency']].corr()

    ggf = GammaGammaFitter(penalizer_coef = 0)
    ggf.fit(returning_customers_summary['frequency'],
            returning_customers_summary['monetary_value'])
    print ggf

    print "Expected conditional average profit: %s, Average profit: %s" % (
        ggf.conditional_expected_average_profit(
            summary['frequency'],
            summary['monetary_value']
        ).mean(),
        summary[summary['frequency']>0]['monetary_value'].mean()
    )

    return ggf.customer_lifetime_value(
        bgf, #the model to use to predict the number of future transactions
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=12, # months
        discount_rate=0.01 # monthly discount rate ~ 12.7% annually
    )