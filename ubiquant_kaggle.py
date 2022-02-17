import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#40 lin mod
CNT_MODELS = 40

#%%time
data = pd.read_csv("../input/ubiquant-market-prediction/train.csv", dtype=dtypes, nrows=500000)

y = data.target.copy()
X = data.drop(columns=['time_id', 'target'])
X.row_id = pd.to_numeric(X.row_id.str.split('_').map(lambda x: x[1]))

models = []

for i in range(0, CNT_MODELS):
    X_s = X.loc[i::CNT_MODELS]
    y_s = y.loc[X_s.index]
    models.append(LinearRegression().fit(X_s, y_s))

class LINEAR_models():
    
    def __init__(self, models):
        self.models = models
        
    def predict(self, X):
        summ = self.models[0].predict(X)
        for model in self.models[1:]:
            summ += model.predict(X)
            
        return summ / len(self.models)
    
model = LINEAR_models(models)

#API for kaggle notebook
import ubiquant
env = ubiquant.make_env()
iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:
    test_df.row_id = pd.to_numeric(test_df.row_id.str.split('_').map(lambda x: x[1]))
    sample_prediction_df['target'] = model.predict(test_df)
    env.predict(sample_prediction_df)
