import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    confusion_matrix, r2_score, f1_score, roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

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
    transaction_data = pd.read_csv(input_file, sep=';') # for FD dataset

    recent_transaction_data = transaction_data[transaction_data['order_date'] > "2014-12-31"]

    # summary = summary_data_from_transaction_data(recent_transaction_data,
    #                                              'customer_id', 'order_date',
    #                                              'revenue', observation_period_end='2016-08-03')
    #
    # print summary.head()

    # summary.to_csv(output_file, sep=';', encoding='utf-8')
else:
    transaction_data = pd.read_csv(input_file, sep=';')

    recent_transaction_data = transaction_data[transaction_data['order_date'] > "2014-12-31"]

    # summary = pd.read_csv(output_file, sep=';')

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

def bgf_fit_in_multiprocessing():
    start_date = date(2016, 3, 20)
    end_date = date(2016, 4, 1)

    periods = [single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)]
    #
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # results = [pool.apply_async(func=cal_vs_holdout_in_parallel, args=(transaction_data, period)).get() for period in periods]


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


def transaction_count_accuracy_calculator(prediction_model, data=recent_transaction_data,
                                          calibration_period_end='2016-05-01',
                                          observation_period_end='2016-08-01'):
    summary_cal_holdout = calibration_and_holdout_data(data, 'customer_id', 'order_date',
                                                       calibration_period_end=calibration_period_end,
                                                       observation_period_end=observation_period_end)

    if prediction_model is None:
        prediction_model = ModifiedBetaGeoFitter(penalizer_coef=0.0)
        prediction_model.fit(summary_cal_holdout['frequency_cal'],
                             summary_cal_holdout['recency_cal'],
                             summary_cal_holdout['T_cal'])

    d_observation = datetime.strptime(observation_period_end, '%Y-%m-%d').date()
    d_calibration = datetime.strptime(calibration_period_end, '%Y-%m-%d').date()

    duration_holdout = (d_observation - d_calibration).days

    summary_cal_holdout['pred_trans_count'] = summary_cal_holdout.\
        apply(lambda r: prediction_model.conditional_expected_number_of_purchases_up_to_time(duration_holdout,
                                                                                             r['frequency_cal'],
                                                                                             r['recency_cal'],
                                                                                             r['T_cal']), axis=1)

    summary_cal_holdout['pred_trans_count'] = summary_cal_holdout['pred_trans_count'].astype(int)

    mse = mean_absolute_error(summary_cal_holdout['frequency_holdout'], summary_cal_holdout['pred_trans_count'])
    mse_div_avg = mean_absolute_error(summary_cal_holdout['frequency_holdout'],
                                      summary_cal_holdout['pred_trans_count']) / \
                  summary_cal_holdout['frequency_holdout'].mean()
    r2 = r2_score(summary_cal_holdout['frequency_holdout'], summary_cal_holdout['pred_trans_count'])

    # return df['real_clv'], df['pred_clv']
    return mse, mse_div_avg, r2


def transaction_count_accuracy_calculator_per_cohort(prediction_model, data=recent_transaction_data, cohort='2015-06',
                                                     calibration_period_end='2016-05-01',
                                                     observation_period_end='2016-08-01'):
    data.set_index('customer_id', inplace=True)
    data['cohort_group'] = data.groupby(level=0)['order_date'].min().apply(lambda x: x.strftime('%Y-%m'))
    data.reset_index(inplace=True)

    data_in_cohort = data[data['cohort_group'] == cohort]

    return transaction_count_accuracy_calculator(prediction_model, data_in_cohort,
                                                 calibration_period_end,
                                                 observation_period_end)


def transaction_count_accuracy_calculator_cohort_comparison(data=recent_transaction_data):
    cohort_start = date(2015, 6, 1)
    cohort_end = date(2016, 3, 1)

    dates = [dt.strftime('%Y-%m') for dt in rrule(MONTHLY, dtstart=cohort_start, until=cohort_end)]

    mse_list = []
    mse_div_avg_list = []
    r2_list = []
    for d in dates:
        mse, mse_div_avg, r2 = transaction_count_accuracy_calculator_per_cohort(None, data, cohort=d)
        mse_list.append(mse)
        mse_div_avg_list.append(mse_div_avg)
        r2_list.append(r2)

    return mse_list, mse_div_avg_list, r2_list


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

    holdout_data = data[(data['order_date'] >= calibration_period_end) & (data['order_date'] <= observation_period_end)]

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

    threshold_1_percent = 0.5
    threshold_2_percent = 0.75

    class_threshold = recent_transaction_data['revenue'].quantile([threshold_1_percent, threshold_2_percent])

    threshold_1 = class_threshold.loc[threshold_1_percent]
    threshold_2 = class_threshold.loc[threshold_2_percent]

    pred_customers_low_value = calibration_summary[calibration_summary['monetary_value'] < threshold_1]
    pred_customers_medium_value = calibration_summary[(calibration_summary['monetary_value'] <= threshold_2) &
                                                      (calibration_summary['monetary_value'] >= threshold_1)]
    pred_customers_high_value = calibration_summary[calibration_summary['monetary_value'] > threshold_2]

    real_customers_low_value = holdout_summary[holdout_summary['monetary_value'] < threshold_1]
    real_customers_medium_value = holdout_summary[(holdout_summary['monetary_value'] <= threshold_2) &
                                                      (holdout_summary['monetary_value'] >= threshold_1)]
    real_customers_high_value = holdout_summary[holdout_summary['monetary_value'] > threshold_2]

    df.ix[pred_customers_low_value.index, 'pred_clv'] = 0
    df.ix[pred_customers_medium_value.index, 'pred_clv'] = 1
    df.ix[pred_customers_high_value.index, 'pred_clv'] = 2

    df.ix[real_customers_low_value.index, 'real_clv'] = 0
    df.ix[real_customers_medium_value.index, 'real_clv'] = 1
    df.ix[real_customers_high_value.index, 'real_clv'] = 2

    cm = confusion_matrix(df['real_clv'], df['pred_clv'])

    f1 = f1_score(df['real_clv'], df['pred_clv'], average='micro')

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
                                 calibration_period_end='2015-08-01', threshold=0.65):
    calibration_data = data[data['order_date'] < calibration_period_end]

    holdout_data = data[data['order_date'] >= calibration_period_end]

    calibration_summary = summary_data_from_transaction_data(calibration_data,
                                                             'customer_id', 'order_date',
                                                             'revenue', observation_period_end=calibration_period_end)

    holdout_summary = summary_data_from_transaction_data(holdout_data,
                                                         'customer_id', 'order_date',
                                                         'revenue', observation_period_end='2016-08-03')

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

    real_customers_not_alive_index = list(set(calibration_summary['frequency'].index) - set(holdout_summary['frequency'].index))

    is_alive = pd.DataFrame(index=calibration_summary['frequency'].index)
    is_alive['real'] = 1
    is_alive['pred'] = 1

    is_alive.ix[real_customers_not_alive_index, 'real'] = 0
    is_alive.ix[pred_customers_not_alive.index, 'pred'] = 0

    cm = confusion_matrix(is_alive['real'], is_alive['pred'])
    float(cm[0][0] + cm[1][1]) / float(len(is_alive.index))

    f1 = f1_score(is_alive['real'], is_alive['pred'])

    return f1


def churning_rate_tp_tn_cutoff_impact(alive_prob, is_alive):
    t = np.arange(0, 1.01, 0.01)

    f1_list = []
    for i in t:
        threshold = i
        pred_customers_not_alive = alive_prob[alive_prob < threshold]

        is_alive['pred'] = 1
        is_alive.ix[pred_customers_not_alive.index, 'pred'] = 0

        f1 = f1_score(is_alive['real'], is_alive['pred'])
        f1_list.append(f1)

    plt.plot(t, f1_list)
    plt.xlabel('Cutoff Threshold for Churn Rate')
    plt.ylabel(r'$F_1$')
    plt.title('Impact of Cutoff Threshold on Churn Rate Prediction Accuracy')
    # plt.axis([0, 1, 0, 1])
    # plt.show()
    save("tp_tn_percentage_threshold_{}".format(plot_source), ext="pdf", close=True, verbose=True)

    max_score = max(f1_list)
    max_index = f1_list.index(max_score)
    optimal_threshold = t[max_index]

    return optimal_threshold

def calibration_period_length_impact(data=recent_transaction_data,
                                     observation_period_start='2014-12-31', observation_period_end='2016-08-01'):
    calibration_percentage = np.arange(0.1, 1, 0.1)

    d_observation_start = datetime.strptime(observation_period_start, '%Y-%m-%d').date()
    d_observation_end = datetime.strptime(observation_period_end, '%Y-%m-%d').date()

    tdelta = (d_observation_end - d_observation_start).days

    f1_list = []

    for cal_percent in calibration_percentage:
        d_calibration_end = datetime.strptime(observation_period_start, '%Y-%m-%d').date()\
                            + timedelta(int(tdelta*cal_percent))
        calibration_period_end = datetime.strftime(d_calibration_end, '%Y-%m-%d')

        f1_result = churning_accuracy_calculator(None, data, calibration_period_end, 0.7)
        f1_list.append(f1_result)

        print "{}% finished".format(cal_percent*100)

    plt.plot(calibration_percentage, f1_list)
    plt.xlabel('Calibration Period Percentage')
    plt.ylabel(r'$F_1$')
    plt.title('Impact of Calibration Period Length on Churn Rate Prediction Accuracy')
    # plt.axis([0, 1, 0, 1])
    # plt.show()
    save("cal_percentage_churnrate_f1_impact_{}".format(plot_source), ext="pdf", close=True, verbose=True)


def churning_accuracy_calculator_with_rf(data=recent_transaction_data, calibration_period_end='2015-08-01'):
    calibration_data = data[data['order_date'] < calibration_period_end]

    holdout_data = data[data['order_date'] >= calibration_period_end]

    calibration_summary = summary_data_from_transaction_data(calibration_data,
                                                             'customer_id', 'order_date',
                                                             'revenue', observation_period_end=calibration_period_end)

    holdout_summary = summary_data_from_transaction_data(holdout_data,
                                                         'customer_id', 'order_date',
                                                         'revenue', observation_period_end='2016-08-03')

    real_customers_not_alive_index = list(
        set(calibration_summary['frequency'].index) - set(holdout_summary['frequency'].index))

    is_alive = pd.DataFrame(index=calibration_summary['frequency'].index)
    is_alive['real'] = 1

    is_alive.ix[real_customers_not_alive_index, 'real'] = 0

    X_train, X_test, y_train, y_test =\
        train_test_split(calibration_summary, is_alive['real'], test_size=0.1, random_state=42)

    clf = RandomForestClassifier(n_estimators=31, max_depth=4)
    clf = clf.fit(X_train, y_train)

    # find the optimal parameter for the classifier - in this case, number of estimators
    # n_estimator_range = range(1, 100)
    # n_scores = []
    # for n in n_estimator_range:
    #     clf = RandomForestClassifier(n_estimators=n)
    #     scores = cross_val_score(clf, calibration_summary, is_alive['real'], cv=10, scoring='accuracy')
    #     n_scores.append(scores.mean())
    # print n_scores
    # % matplotlib inline
    # plt.plot(n_estimator_range, n_scores)

    # find the optimal parameter for the classifier - in this case, max tree depth

    # n_depth_range = range(1, 15)
    # n_scores = []
    # for depth in n_depth_range:
    #     clf = RandomForestClassifier(n_estimators=10, max_depth=depth)
    #     scores = cross_val_score(clf, calibration_summary, is_alive['real'], cv=10, scoring='f1')
    #     print scores.mean()
    #     n_scores.append(scores.mean())
    # print n_scores
    #
    # plt.plot(n_depth_range, n_scores)
    # plt.xlabel('Max Tree Depth')
    # plt.ylabel(r'$F_1$')
    # plt.title('Impact of Tree Depth on Churn Rate Prediction Accuracy')
    # # plt.axis([0, 1, 0, 1])
    # # plt.show()
    # save("tree_depth_avg_churnrate_f1_impact_{}".format(plot_source), ext="pdf", close=True, verbose=True)

    # draw roc curve for the classifier
    # fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)
    # plt.plot(fpr, tpr)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)

    return cm, f1, auc(fpr, tpr)


def random_forest_optimizer(X, y, scoring='f1', n_iter=10):
    n_estimators_range = list(range(1, 51))
    max_depth_range = list(range(1, 16))

    # create a parameter grid: map the parameter names to the values that should be searched
    param_dist = dict(n_estimators=n_estimators_range, max_depth=max_depth_range)
    # print(param_grid)

    rfc = RandomForestClassifier()

    # grid = GridSearchCV(rfc, param_grid, cv=10, scoring=scoring, n_jobs=-1)
    # grid.fit(X, y)

    best_scores = []
    best_parameters = []
    for _ in range(20):
        rand = RandomizedSearchCV(rfc, param_dist, cv=10, scoring=scoring, n_iter=n_iter, n_jobs=-1)
        rand.fit(X, y)
        best_scores.append(rand.best_score_)
        best_parameters.append(rand.best_params_)

    max_score = max(best_scores)
    max_index = best_scores.index(max_score)
    optimal_parameter = best_parameters[max_index]

    return optimal_parameter


def knn_optimizer(X, y, scoring='f1', n_iter=10):
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    # create a parameter grid: map the parameter names to the values that should be searched
    param_dist = dict(n_neighbors=k_range, weights=weight_options)
    # print(param_grid)

    knn = KNeighborsClassifier()

    # grid = GridSearchCV(rfc, param_grid, cv=10, scoring=scoring, n_jobs=-1)
    # grid.fit(X, y)

    best_scores = []
    best_parameters = []
    for _ in range(20):
        rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring=scoring, n_iter=n_iter, n_jobs=-1)
        rand.fit(X, y)
        best_scores.append(rand.best_score_)
        best_parameters.append(rand.best_params_)

    max_score = max(best_scores)
    max_index = best_scores.index(max_score)
    optimal_parameter = best_parameters[max_index]

    return optimal_parameter


def churning_accuracy_calculator_with_xgboost(data=recent_transaction_data, calibration_period_end='2016-03-01'):
    calibration_data = data[data['order_date'] < calibration_period_end]

    holdout_data = data[data['order_date'] >= calibration_period_end]

    calibration_summary = summary_data_from_transaction_data(calibration_data,
                                                             'customer_id', 'order_date',
                                                             'revenue', observation_period_end=calibration_period_end)

    holdout_summary = summary_data_from_transaction_data(holdout_data,
                                                         'customer_id', 'order_date',
                                                         'revenue', observation_period_end='2016-08-03')

    calibration_summary = feature_adder(calibration_data, calibration_summary, calibration_period_end)

    real_customers_not_alive_index = list(
        set(calibration_summary['frequency'].index) - set(holdout_summary['frequency'].index))

    is_alive = pd.DataFrame(index=calibration_summary['frequency'].index)
    is_alive['real'] = 1

    is_alive.ix[real_customers_not_alive_index, 'real'] = 0

    predictors = [x for x in calibration_summary.columns if x not in ['customer_id']]

    X_train, X_test, y_train, y_test = \
        train_test_split(calibration_summary, is_alive['real'], test_size=0.3, random_state=42)

    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    xgboost_fit(xgb1, X_train, X_test, y_train, y_test, predictors)

    # test-auc reachs optimum when n_estimators = 140/255

    # Grid seach on max_depth and min_child_weight
    # Choose all predictors except target & IDcols
    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch1.fit(X_train[predictors], y_train)
    # gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

    # test-auc reachs optimum when {'max_depth': 3, 'min_child_weight': 5}

    # Grid seach on max_depth and min_child_weight
    # Choose all predictors except target & IDcols
    param_test2 = {
        'max_depth': [2, 3, 4],
        'min_child_weight': [4, 5, 6]
    }
    gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                    min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test2, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch2.fit(X_train[predictors], y_train)
    # gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

    # test-auc reachs optimum when {'max_depth': 4/3, 'min_child_weight': 4/3}

    # Grid seach on min_child_weight
    # Choose all predictors except target & IDcols
    param_test2b = {
        'min_child_weight': [2, 3, 4, 5, 6]
    }
    gsearch2b = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
                                                     min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                     objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                     seed=27),
                             param_grid=param_test2b, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch2b.fit(X_train[predictors], y_train)
    # gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

    # test-auc reachs optimum when {'min_child_weight': 3}

    # Grid seach on gamma
    # Choose all predictors except target & IDcols
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
                                                    min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test3, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch3.fit(X_train[predictors], y_train)
    # gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

    # test-auc reachs optimum when {'gamma': 0.0/0.3}

    xgb2 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=3,
        min_child_weight=3,
        gamma=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    xgboost_fit(xgb2, X_train, X_test, y_train, y_test, predictors)

    # test-auc reachs optimum when n_estimators = 253/209

    # Grid seach on subsample and max_features
    # Choose all predictors except target & IDcols
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=253, max_depth=4,
                                                    min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test4, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch4.fit(X_train[predictors], y_train)
    # gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

    # test-auc reachs optimum when {'colsample_bytree': 0.7, 'subsample': 0.9}

    # Grid seach on subsample
    # Choose all predictors except target & IDcols
    param_test4a = {
        'subsample': [i / 10.0 for i in range(9, 11)]
    }
    gsearch4a = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=253, max_depth=4,
                                                    min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.7,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test4a, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch4a.fit(X_train[predictors], y_train)
    # gsearch4a.grid_scores_, gsearch4a.best_params_, gsearch4a.best_score_

    # test-auc reachs optimum when {'subsample': 0.9}

    # Grid seach on subsample and max_features
    # Choose all predictors except target & IDcols
    param_test5 = {
        'subsample': [i / 100.0 for i in range(85, 100, 5)],
        'colsample_bytree': [i / 100.0 for i in range(65, 80, 5)]
    }
    gsearch5 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=253, max_depth=4,
                                                    min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test5, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch5.fit(X_train[predictors], y_train)
    # gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

    # test-auc reachs optimum when {'colsample_bytree': 0.7, 'subsample': 0.9}

    # Grid seach on reg_alpha
    # Choose all predictors except target & IDcols
    param_test6 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 10, 100]
    }
    gsearch6 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=253, max_depth=4,
                                                    min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test6, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch6.fit(X_train[predictors], y_train)
    # gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

    # test-auc reachs optimum when {'reg_alpha': 10}

    # Grid seach on reg_alpha
    # Choose all predictors except target & IDcols
    param_test7 = {
        'reg_alpha': [0, 2.5, 5, 10, 50]
    }
    gsearch7 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=253, max_depth=4,
                                                    min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test7, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)
    gsearch7.fit(X_train[predictors], y_train)
    # gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_

    # test-auc reachs optimum when {'reg_alpha': 5}

    xgb3 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=4,
        min_child_weight=3,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_alpha=5,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    xgboost_fit(xgb3, X_train, X_test, y_train, y_test, predictors)

    # test-auc reachs optimum when n_estimators = 246

    xgb4 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=4,
        min_child_weight=3,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_alpha=5,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    xgboost_fit(xgb4, X_train, X_test, y_train, y_test, predictors)

    # optimal XGB for dataset without column 'monetary_value'
    # xgb_final = XGBClassifier(
    #     learning_rate=0.01,
    #     n_estimators=1846,
    #     max_depth=4,
    #     min_child_weight=3,
    #     gamma=0,
    #     subsample=0.9,
    #     colsample_bytree=0.7,
    #     reg_alpha=5,
    #     objective='binary:logistic',
    #     nthread=4,
    #     scale_pos_weight=1,
    #     seed=27)
    #
    # xgboost_fit(xgb_final, X_train, X_test, y_train, y_test, predictors)

    # optimal XGB for dataset with column 'monetary_value'
    xgb_final = XGBClassifier(
        learning_rate=0.1,
        n_estimators=288,
        max_depth=3,
        min_child_weight=3,
        gamma=0.3,
        subsample=0.7,
        colsample_bytree=0.75,
        reg_alpha=10,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    xgboost_fit(xgb_final, X_train, X_test, y_train, y_test, predictors)

    # print xgb_final.feature_importances_

    dtest_predictions = xgb4.predict(X_test[predictors])
    dtest_predprob = xgb4.predict_proba(X_test[predictors])[:, 1]

    f1 = f1_score(y_test, dtest_predictions)
    cm = confusion_matrix(y_test, dtest_predictions)

    fpr, tpr, thresholds = roc_curve(y_test, dtest_predprob, pos_label=1)
    plt.plot(fpr, tpr)
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('ROC Curve')
    save("roc_curve_churnrate_{}".format(plot_source), ext="pdf", close=True, verbose=True)


def xgboost_fit(alg, X_train, X_test, y_train, y_test,
                predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train[predictors].values, label=y_train.values)
        xgtest = xgb.DMatrix(X_test[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X_train[predictors], y_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(X_train[predictors])
    dtrain_predprob = alg.predict_proba(X_train[predictors])[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % accuracy_score(y_train.values, dtrain_predictions)
    print "AUC Score (Train): %f" % roc_auc_score(y_train, dtrain_predprob)

    #     Predict on testing data:
    dtest_predprob = alg.predict_proba(X_test[predictors])[:, 1]
    print 'AUC Score (Test): %f' % roc_auc_score(y_test, dtest_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


def feature_adder(calibration_data, calibration_summary, calibration_period_end):
    d_calibration = datetime.strptime(calibration_period_end, '%Y-%m-%d').date()
    d_point1 = d_calibration - timedelta(days=7)
    calibration_point1 = calibration_data[(calibration_data.order_date > d_point1.strftime('%Y-%m-%d')) & (
        calibration_data.order_date < calibration_period_end)].groupby('customer_id', sort=False)['order_date'].agg(
        ['count'])

    calibration_summary['recent_week_transaction_count'] = 0
    calibration_summary.ix[calibration_point1.index, 'recent_week_transaction_count'] = calibration_point1['count']

    d_point2 = d_calibration - timedelta(days=30)
    calibration_point2 = calibration_data[(calibration_data.order_date > d_point2.strftime('%Y-%m-%d')) & (
        calibration_data.order_date < calibration_period_end)].groupby('customer_id', sort=False)['order_date'].agg(
        ['count'])

    calibration_summary['recent_month_transaction_count'] = 0
    calibration_summary.ix[calibration_point2.index, 'recent_month_transaction_count'] = calibration_point2['count']

    d_point3 = d_calibration - timedelta(days=180)
    calibration_point3 = calibration_data[(calibration_data.order_date > d_point3.strftime('%Y-%m-%d')) & (
        calibration_data.order_date < calibration_period_end)].groupby('customer_id', sort=False)['order_date'].agg(
        ['count'])

    calibration_summary['recent_six_months_transaction_count'] = 0
    calibration_summary.ix[calibration_point3.index, 'recent_six_months_transaction_count'] = calibration_point3['count']

    return calibration_summary

print "churn rate prediction begins..."

cm, f1 = churning_accuracy_calculator_with_rf(recent_transaction_data, '2015-08-01')

print "confusion matrix:"
print cm

print 'F1={}'.format(f1)