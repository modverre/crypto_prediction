import numpy as np
from datetime import datetime
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler

def get_X_y(history_size, dataset):
    '''function that splits train / test sets in X and y'''

    X = []
    y = []

    for i in range(history_size, dataset.shape[0]):
        X.append(dataset[i-history_size:i,:])
        y.append(dataset[i,0])

    return np.array(X), np.array(y)


def inverse_transformer(y, scaler):
    '''function that takes a one-dimensional input array (y_test or y_hat) and inverse transforms it.'''
    y = np.c_[y, np.ones(len(y))]

    y = scaler.inverse_transform(y)

    y= y[:,0]

    return y

def preprocess_prediction(df):
    """method that pre-process the data for prediction"""

    # log transforming the data
    df["high"] = np.log(df["high"])

    # instantiating the scaler
    scaler = joblib.load('scaler.joblib')

    # selecting relevant column from df
    dataset = df.values

    # scaling the data
    dataset_scaled = scaler.transform(dataset)

    dataset_scaled = dataset_scaled.reshape(1,dataset_scaled.shape[0],dataset_scaled.shape[1])

    return dataset_scaled, scaler

def inverse_scale_prediction(pred):

    scaler = joblib.load('scaler.joblib')

    pred = inverse_transformer(pred, scaler)

    pred = np.exp(pred)

    return pred


def date2utc_ts(date):
    """ transforms utc-timestring 2021-11-24T10:00:00Z to unix-timestamp"""
    date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
    return int(date.replace(tzinfo=datetime.timezone.utc).timestamp())


def gecko_make_df(raw):
    """
    puts the weird list from coingecko into a neat dataframe
    output:
        dataframe - datetime(index), prices, market caps, total_volumes
    """

    df = pd.DataFrame(raw['prices'], columns=['timestamp', 'price'])
    df2 = pd.DataFrame(raw['market_caps'], columns=['ts', 'market_caps'])
    df3 = pd.DataFrame(raw['total_volumes'], columns=['ts', 'total_volumes'])
    df['market_caps'] = df2['market_caps']
    df['total_volumes'] = df3['total_volumes']
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')
    return df
