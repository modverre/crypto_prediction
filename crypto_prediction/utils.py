import numpy as np
import datetime as datetime
import pandas as pd
import joblib
import os

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
    scaler = joblib.load('crypto_prediction/scaler.joblib')

    # selecting relevant column from df
    dataset = df.values

    # scaling the data
    dataset_scaled = scaler.transform(dataset)

    dataset_scaled = dataset_scaled.reshape(1,dataset_scaled.shape[0],dataset_scaled.shape[1])

    return dataset_scaled

def inverse_scale_prediction(pred):

    scaler = joblib.load('crypto_prediction/scaler.joblib')

    pred = inverse_transformer(pred, scaler)

    pred = np.exp(pred)

    return pred


def date2utc_ts(date):
    """ transforms utc-timestring 2021-11-24T10:00:00Z to unix-timestamp"""
    date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
    return int(date.replace(tzinfo=datetime.timezone.utc).timestamp())


def gecko_make_df(raw):
    """
    transforms the weird list from coingecko into a neat dataframe we all cherish
    input:
        hourly, smaller values are mean'ed to the full hour

    output:
        dataframe - datetime(index), timestamp, price, market_caps, total_volumes
    """
    # make the 'normal' dataframe
    df = pd.DataFrame(raw['prices'], columns=['timestamp', 'price'])
    df2 = pd.DataFrame(raw['market_caps'], columns=['ts', 'market_caps'])
    df3 = pd.DataFrame(raw['total_volumes'], columns=['ts', 'total_volumes'])
    df['market_caps'] = df2['market_caps']
    df['total_volumes'] = df3['total_volumes']
    df['datetime_original'] = pd.to_datetime(df['timestamp'], unit='ms')

    # set the minutes and less to zero, 2021-10-28 08:03:46.436 becomes 2021-10-28 08:00:00
    df['datetime'] = df['datetime_original'].apply(lambda x: x.replace(minute=0, second=0, microsecond=0))

    # set the new index to be able to group the data
    df = df.set_index('datetime') # works even with same values as datetime

    # group hourly and take the mean - if only 1 value per hour its fine already
    df = df.groupby(pd.Grouper(freq='H')).mean()
    df = pd.DataFrame(df)

    return df

def twitter_make_df(raw):
    df = pd.DataFrame.from_dict(raw)
    # drop last hour since its incomplete
    df = df.iloc[:-1]
    # 2021-11-30T13:12.164Z to 2021-11-30 13:12
    df['datetime'] = pd.to_datetime(df['end'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    # cut 2021-11-30 13:12 to 2021-11-30 13:00
    df['datetime'] = df['datetime'].apply(lambda x: x.replace(minute=0, second=0, microsecond=0))
    # set datetime as index
    df = df.set_index('datetime')
    # drop the rest
    df.drop(['start', 'end'], axis=1, inplace=True)

    return df
