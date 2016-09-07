import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, r2_score, f1_score

import matplotlib.pyplot as plt

from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data, customer_lifetime_value
from lifetimes import BetaGeoFitter, GammaGammaFitter, ModifiedBetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix, \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases, \
    plot_history_alive

from multiprocessing import Process, Queue, current_process, cpu_count

from datetime import timedelta, date, datetime
from dateutil.rrule import rrule, MONTHLY
from plt_save import save


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

is_summary_available = True
input_file = "sg_customers.csv"
output_file = "summary_sg_customers.csv"

calibration_output_file = "summary_calibration_sg_customers.csv"
holdout_output_file = "summary_holdout_sg_customers.csv"

plot_source = "sg"

if is_summary_available is False:
    # transaction_data = pd.read_csv(input_file, sep=';')
    transaction_data = pd.read_excel(input_file, sep=';') # for FD dataset

    recent_transaction_data = transaction_data[transaction_data['order_date'] > "2014-12-31"]

    summary = summary_data_from_transaction_data(recent_transaction_data,
                                                 'customer_id', 'order_date',
                                                 'revenue', observation_period_end='2016-08-03')

    print summary.head()

    summary.to_csv(output_file, sep=';', encoding='utf-8')
else:
    transaction_data = pd.read_csv(input_file, sep=';')

    recent_transaction_data = transaction_data[transaction_data['order_date'] > "2014-12-31"]

    summary = pd.read_csv(output_file, sep=';')

if plot_source is "sg":
    recent_transaction_data['order_date'] = recent_transaction_data.order_date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))


def bgf_analyser(summary, plot_source):
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary['frequency'],
            summary['recency'],
            summary['T'])

    plot_frequency_recency_matrix(bgf)
    save("f_r_matrix_{}".format(plot_source), ext="pdf", close=True, verbose=True)

    plot_probability_alive_matrix(bgf)
    save("pr_alive_matrix_{}".format(plot_source), ext="pdf", close=True, verbose=True)

    plot_period_transactions(bgf)
    save("period_transactions_{}".format(plot_source), ext="pdf", close=True, verbose=True)

    return bgf


def bgf_history_alive(data=recent_transaction_data, bgf=None, customer_id='80_3811'):
    id = customer_id
    days_since_birth = 365
    sp_trans = data.ix[data['customer_id'] == id]
    plot_history_alive(bgf, days_since_birth, sp_trans, 'order_date')
    save("history_alive_{}".format(plot_source), ext="pdf", close=True, verbose=True)


def cal_vs_holdout_in_parallel(data=recent_transaction_data, calibration_period_end='2016-05-01'):
    print calibration_period_end

    bgf_ = BetaGeoFitter(penalizer_coef=0.0)
    summary_cal_holdout = calibration_and_holdout_data(data, 'customer_id', 'order_date',
                                                           calibration_period_end=calibration_period_end,
                                                           observation_period_end='2016-08-03')
    print summary_cal_holdout.head()
    bgf_.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])

    # plot_calibration_purchases_vs_holdout_purchases(bgf_, summary_cal_holdout)
    # save("cal_vs_holdout_{}".format(calibration_period_end), ext="pdf", close=True, verbose=True)

start_date = date(2016, 3, 20)
end_date = date(2016, 4, 1)

periods = [single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)]
#
# pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
# results = [pool.apply_async(func=cal_vs_holdout_in_parallel, args=(transaction_data, period)).get() for period in periods]

#
# Function run by worker processes
#

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)

#
# Function used to calculate result
#

def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % \
        (current_process().name, func.__name__, args, result)

#
# Functions referenced by tasks
#

NUMBER_OF_PROCESSES = cpu_count()
TASKS1 = [(cal_vs_holdout_in_parallel, (transaction_data, period)) for period in periods]

# Create queues
task_queue = Queue()
done_queue = Queue()

# Submit tasks
for task in TASKS1:
    task_queue.put(task)

# Start worker processes
for i in range(NUMBER_OF_PROCESSES):
    Process(target=worker, args=(task_queue, done_queue)).start()

# Get and print results
print 'Unordered results:'
for i in range(len(TASKS1)):
    print '\t', done_queue.get()

# Tell child processes to stop
for i in range(NUMBER_OF_PROCESSES):
    task_queue.put('STOP')


def ggf_analyser(summary, bgf=None):
    if bgf is None:
        bgf = BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(summary['frequency'],
                summary['recency'],
                summary['T'])

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

    return customer_lifetime_value(
        bgf, #the model to use to predict the number of future transactions
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=12, # months
        discount_rate=0.01 # monthly discount rate ~ 12.7% annually
    )


def clv_accuracy_calculator(prediction_model, data=recent_transaction_data,
                            discount_rate=0, calibration_period_end='2016-05-01',
                            observation_period_end='2016-08-01'):
    calibration_data = data[data['order_date'] < calibration_period_end]

    holdout_data = data[data['order_date'] >= calibration_period_end]

    calibration_summary = summary_data_from_transaction_data(calibration_data,
                                                 'customer_id', 'order_date',
                                                 'revenue', observation_period_end=calibration_period_end)

    holdout_summary = summary_data_from_transaction_data(holdout_data,
                                                 'customer_id', 'order_date',
                                                 'revenue', observation_period_end='2016-08-03')

    print calibration_summary.head()
    print holdout_summary.head()

    calibration_summary.to_csv(calibration_output_file, sep=';', encoding='utf-8')
    holdout_summary.to_csv(holdout_output_file, sep=';', encoding='utf-8')

    if prediction_model is None:
        prediction_model = ModifiedBetaGeoFitter(penalizer_coef=0.0)
        prediction_model.fit(calibration_summary['frequency'],
                             calibration_summary['recency'],
                             calibration_summary['T'])

    d_observation = datetime.strptime(observation_period_end, '%Y-%m-%d').date()
    d_calibration = datetime.strptime(calibration_period_end, '%Y-%m-%d').date()

    df = pd.DataFrame(index=calibration_summary['frequency'].index)
    df['pred_clv'] = 0  # initialize the pred_clv column to zeros
    df['real_clv'] = 0

    for i in range(30, ((d_observation-d_calibration).days/30)*30+1, 30):
        expected_number_of_transactions = prediction_model.predict(i, calibration_summary['frequency'],
                                                                   calibration_summary['recency'],
                                                                   calibration_summary['T']) - \
                                          prediction_model.predict(i-30, calibration_summary['frequency'],
                                                                   calibration_summary['recency'],
                                                                   calibration_summary['T'])
        # sum up the CLV estimates of all of the periods
        df['pred_clv'] += (calibration_summary['monetary_value'] * expected_number_of_transactions) / \
                          (1 + discount_rate) ** (i / 30)

    df['real_clv'] = holdout_data.groupby('customer_id')['revenue'].sum()

    df['real_clv'] = df['real_clv'].fillna(0)

    mse = mean_squared_error(df['real_clv'], df['pred_clv'])
    mse_div_avg = mean_squared_error(df['real_clv'], df['pred_clv'])/df['real_clv'].mean()
    r2 = r2_score(df['real_clv'], df['pred_clv'])

    # return df['real_clv'], df['pred_clv']
    return mse, mse_div_avg, r2


def clv_classifier(data=recent_transaction_data,
                   calibration_period_end='2016-05-01',
                   observation_period_end='2016-08-01'):
    calibration_data = data[data['order_date'] < calibration_period_end]

    holdout_data = data[data['order_date'] >= calibration_period_end]

    calibration_summary = summary_data_from_transaction_data(calibration_data,
                                                             'customer_id', 'order_date',
                                                             'revenue', observation_period_end=calibration_period_end)

    holdout_summary = summary_data_from_transaction_data(holdout_data,
                                                         'customer_id', 'order_date',
                                                         'revenue', observation_period_end=observation_period_end)

    is_alive_index = list(set(calibration_summary['frequency'].index) | set(holdout_summary['frequency'].index))

    df = pd.DataFrame(index=is_alive_index)

    df['pred_clv'] = 0  # initialize the pred_clv column to zeros
    df['real_clv'] = 0

    pred_customers_low_value = calibration_summary[calibration_summary['monetary_value'] < 7.27]
    pred_customers_medium_value = calibration_summary[(calibration_summary['monetary_value'] <= 9.38) &
                                                      (calibration_summary['monetary_value'] >= 7.27)]
    pred_customers_high_value = calibration_summary[calibration_summary['monetary_value'] > 9.38]

    real_customers_low_value = holdout_summary[holdout_summary['monetary_value'] < 7.27]
    real_customers_medium_value = holdout_summary[(holdout_summary['monetary_value'] <= 9.38) &
                                                      (holdout_summary['monetary_value'] >= 7.27)]
    real_customers_high_value = holdout_summary[holdout_summary['monetary_value'] > 9.38]

    df.ix[pred_customers_low_value.index, 'pred_clv'] = 0
    df.ix[pred_customers_medium_value.index, 'pred_clv'] = 1
    df.ix[pred_customers_high_value.index, 'pred_clv'] = 2

    df.ix[real_customers_low_value.index, 'real_clv'] = 0
    df.ix[real_customers_medium_value.index, 'real_clv'] = 1
    df.ix[real_customers_high_value.index, 'real_clv'] = 2

    cm = confusion_matrix(df['real_clv'], df['pred_clv'])

    f1 = f1_score(df['real_clv'], df['pred_clv'])

    return cm, f1


def clv_accuracy_calculator_per_cohort(prediction_model, data=recent_transaction_data, cohort='2015-06',
                                       discount_rate=0, calibration_period_end='2016-05-01',
                                       observation_period_end='2016-08-01'):
    data.set_index('customer_id', inplace=True)
    data['cohort_group'] = data.groupby(level=0)['order_date'].min().apply(lambda x: x.strftime('%Y-%m'))
    data.reset_index(inplace=True)

    data_in_cohort = data[data['cohort_group'] == cohort]

    return clv_accuracy_calculator(prediction_model, data_in_cohort,
                                   discount_rate, calibration_period_end,
                                   observation_period_end)


def clv_accuracy_calculator_cohort_comparison(data=recent_transaction_data):
    cohort_start = date(2015, 6, 1)
    cohort_end = date(2016, 3, 1)

    dates = [dt.strftime('%Y-%m') for dt in rrule(MONTHLY, dtstart=cohort_start, until=cohort_end)]

    mse_list = []
    mse_div_avg_list = []
    r2_list = []
    for d in dates:
        mse, mse_div_avg, r2 = clv_accuracy_calculator_per_cohort(None, data, cohort=d)
        mse_list.append(mse)
        mse_div_avg_list.append(mse_div_avg)
        r2_list.append(r2)

    return mse_list, mse_div_avg_list, r2_list


def churning_accuracy_calculator(prediction_model, data=recent_transaction_data,
                                 calibration_period_end='2015-08-01', threshold=0.01):
    calibration_data = data[data['order_date'] < calibration_period_end]

    holdout_data = data[data['order_date'] >= calibration_period_end]

    calibration_summary = summary_data_from_transaction_data(calibration_data,
                                                             'customer_id', 'order_date',
                                                             'revenue', observation_period_end=calibration_period_end)

    holdout_summary = summary_data_from_transaction_data(holdout_data,
                                                         'customer_id', 'order_date',
                                                         'revenue', observation_period_end='2016-08-03')

    print calibration_summary.head()
    print holdout_summary.head()

    calibration_summary.to_csv(calibration_output_file, sep=';', encoding='utf-8')
    holdout_summary.to_csv(holdout_output_file, sep=';', encoding='utf-8')

    if prediction_model is None:
        prediction_model = ModifiedBetaGeoFitter(penalizer_coef=0.0)
        prediction_model.fit(calibration_summary['frequency'],
                             calibration_summary['recency'],
                             calibration_summary['T'])

    alive_prob = calibration_summary.apply(lambda row:
                                           prediction_model.conditional_probability_alive(
                                               row['frequency'], row['recency'], row['T']), axis=1)

    # customers = holdout_data.groupby('customer_id', sort=False)['order_date'].agg(['count'])
    #
    # real_customers_alive = customers[customers['count'] > 0]
    pred_customers_not_alive = alive_prob[alive_prob < threshold]

    is_alive_index = list(set(calibration_summary['frequency'].index) | set(holdout_summary['frequency'].index))
    real_customers_not_alive_index = list(set(calibration_summary['frequency'].index) - set(holdout_summary['frequency'].index))

    is_alive = pd.DataFrame(index=is_alive_index)
    is_alive['real'] = 1
    is_alive['pred'] = 1

    is_alive.ix[real_customers_not_alive_index, 'real'] = 0
    is_alive.ix[pred_customers_not_alive.index, 'pred'] = 0

    cm = confusion_matrix(is_alive['real'], is_alive['pred'])
    float(cm[0][0] + cm[1][1]) / float(len(is_alive.index))

    f1 = f1_score(is_alive['real'], is_alive['pred'])

    return alive_prob, is_alive


def churning_rate_tp_tn_cutoff_impact(alive_prob, is_alive):
    t = np.arange(0, 1.01, 0.01)

    tp_tn_list = []
    for i in t:
        threshold = i
        pred_customers_not_alive = alive_prob[alive_prob < threshold]

        is_alive['pred'] = 1
        is_alive.ix[pred_customers_not_alive.index, 'pred'] = 0

        cm = confusion_matrix(is_alive['real'], is_alive['pred'])
        tp_tn_list.append(float(cm[0][0] + cm[1][1]) / float(len(is_alive.index)))

    plt.plot(t, tp_tn_list)
    plt.xlabel('Cutoff Threshold for Churn Rate')
    plt.ylabel('Percentage of TP+TN')
    plt.title('Impact of Cutoff Threshold on Churn Rate Prediction Accuracy')
    # plt.axis([0, 1, 0, 1])
    # plt.show()
    save("tp_tn_percentage_threshold_{}".format(plot_source), ext="pdf", close=True, verbose=True)
