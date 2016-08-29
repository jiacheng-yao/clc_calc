import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from plt_save import save

import seaborn as sns

plot_source = "fd"

pd.set_option('max_columns', 50)
mpl.rcParams['lines.linewidth'] = 2

df = pd.read_excel('FD DE customer orders and revenue (1).xlsx')
df.head()

df['order_period'] = df.order_date.apply(lambda x: x.strftime('%Y-%m'))
df.head()

df.set_index('customer_id', inplace=True)
df['cohort_group'] = df.groupby(level=0)['order_date'].min().apply(lambda x: x.strftime('%Y-%m'))
df.reset_index(inplace=True)
df.head()

grouped = df.groupby(['cohort_group', 'order_period'])

cohorts = grouped.agg({'customer_id': pd.Series.nunique,
                       'order_date': pd.Series.nunique,
                       'revenue': np.sum})

cohorts.rename(columns={'customer_id': 'total_users',
                        'order_date': 'total_orders'}, inplace=True)

cohorts.total_orders = cohorts.total_orders.astype('int')

cohorts.head()


def cohort_period(df):
    """
    Creates a `CohortPeriod` column, which is the Nth period based on the user's first purchase.

    Example
    -------
    Say you want to get the 3rd month for every user:
        df.sort(['UserId', 'OrderTime', inplace=True)
        df = df.groupby('UserId').apply(cohort_period)
        df[df.CohortPeriod == 3]
    """
    df['cohort_period'] = np.arange(len(df)) + 1
    return df


cohorts = cohorts.groupby(level=0).apply(cohort_period)
cohorts.head()

cohorts.reset_index(inplace=True)
cohorts.set_index(['cohort_group', 'cohort_period'], inplace=True)

# create a Series holding the total size of each CohortGroup
cohort_group_size = cohorts['total_users'].groupby(level=0).first()
cohort_group_size.head()

user_retention = cohorts['total_users'].unstack(0).divide(cohort_group_size, axis=1)
user_retention.head(10)

user_retention[['2015-06', '2015-07', '2015-08',
                '2015-09', '2015-10', '2015-11',
                '2015-12', '2016-01', '2016-02',
                '2016-03', '2016-04', '2016-05']].plot(figsize=(10,5))
plt.title('Cohorts: User Retention')
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel('% of Cohort Purchasing');

save("user_retention_{}".format(plot_source), ext="pdf", close=True, verbose=True)

sns.set(style='white')

plt.figure(figsize=(12, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%')
save("user_retention_heatmap_{}".format(plot_source), ext="pdf", close=True, verbose=True)
