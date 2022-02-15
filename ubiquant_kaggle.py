import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost
from xgboost import XGBRegressor
import ubiquant
#30 lin mod
CNT_MODELS = 30


data = pd.read_csv('data.csv')

y = data.pop('target')
X = data.drop(columns=['row_id'])

models = []

for i in range(0, CNT_MODELS):
    X_s = X.loc[i::CNT_MODELS]
    y_s = y.loc[X_s.index]
    models.append(XGBRegressor(tree_method = 'gpu_hist', gpu_id = 0).fit(X_s, y_s))

df = pd.DataFrame(columns=np.arange(len(models)))

for i, model in enumerate(models):
    df[i]=model.predict(X[:1000000])

regr = XGBRegressor(tree_method='gpu_hist', gpu_id=0).fit(df.values, y[:1000000])

env = ubiquant.make_env()
iter_test = env.iter_test()
i = 0

for (test_df, sample_prediction_df) in iter_test:
    test_df.reset_index(inplace = True)
    test_df.pop('row_id')
    test_df.rename(columns={'index':'time_id'}, inplace = True)
    test_df['time_id'] = i
    print(test_df)
    df = pd.DataFrame(columns=np.arange(len(models)))
    for i, model in enumerate(models):
        df[i] = model.predict(test_df)
    sample_prediction_df['target'] = regr.predict(df)
    env.predict(sample_prediction_df)
    i += 1
