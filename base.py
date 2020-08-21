import pandas as pd
import numpy as np

from sklearn import metrics
from m5.helper import train_test_split, build_df
import lightgbm as lgb

# os.chdir('c:/users/kk/pycharmprojects/ml/m5')

# Data load-in
# ==========================
df = pd.read_csv('./data/sales_train_evaluation.csv')
calendar = pd.read_csv('./data/calendar.csv')
price = pd.read_csv('./data/sell_prices.csv')


# Preprocess
# ==========================
# Extract just 'CA_2' for fitting.
CA_2_df = build_df(df, calendar, price, 'CA_2')
CA_2_df = CA_2_df[['item_id', 'd', 'demand', 'weekday', 'month',
                  'event_name_1', 'event_name_2', 'snap_CA']]

# Set the size of train and validation set.
train, val = train_test_split(CA_2_df, 1800, 28)

X_train, Y_train = train.drop(['d', 'demand'], axis=1), train['demand']
X_val, Y_val = val.drop(['d', 'demand'], axis=1), val['demand']


# Fitting
# ===========================
params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'n_jobs': -1,
    'seed': 999,
    'learning_rate': 0.05,
    'bagging_fraction': 0.75,
    'bagging_freq': 10,
    'feature_fraction': 0.75,
}

reg = lgb.LGBMRegressor(**params, n_estimators=3000)
reg.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], early_stopping_rounds=50, verbose=500)

# As can be noticed, simply putting explanatory variables in a lgbm isn't satisfactory.
pred_val = reg.predict(X_val, num_iteration=reg.best_iteration_)
score = np.sqrt(metrics.mean_squared_error(pred_val, Y_val))
