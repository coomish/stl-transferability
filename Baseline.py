import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from utils import mape


def baseline(df):
    # predict next year by just taking the values from last year
    predicted = df.loc['2016-01-01':'2016-12-31']
    actual = df.loc['2017-01-01':'2017-12-31']
    # and padding the missing values with the last datapoint
    actual.loc[actual < 1] = np.nan
    actual = actual.fillna(method='pad')
    # transform df to series
    predicted = pd.Series(predicted)
    actual = pd.Series(actual)
    print("Number of actual days:", len(actual))
    print('Number of predicted days:', len(predicted))
    rmse = sqrt(mean_squared_error(actual, predicted))
    print('For 12 Months RMSE: %.3f' % rmse)
    mape_value = mape(actual, predicted)
    print ("For 12 Months MAPE :", mape_value)


def calculate_baseline(path):
    df = pd.read_pickle(path)
    baseline(df)

# Call Baseline Method
calculate_baseline("data/preprocessed/branch1.pkl")
