import numpy as np
import datetime as datetime
import pandas as pd
import joblib
import os

from sklearn.preprocessing import MinMaxScaler

horizon = 24
coins = 19
list_of_dfs = ["ban", "cummies", "dinu", "doge",
"doggy", "elon", "erc20", "ftm", "grlc", "hoge",
"lowb", "mona", "samo", "shib", "shibx", "smi",
"wow", "yooshi","yummy"]

def get_X_y(history_size, horizon, dataset):

    X = []
    y = []

    for i in range(0, dataset.shape[0]-history_size-horizon):
        X.append(dataset[i:i+history_size,:])
        y.append(dataset[i+history_size: i+history_size+horizon,0])

    return np.array(X), np.array(y)


def inverse_transformer(y, scaler):

  # inverse scaling
  y = np.c_[y, 1,1,1,1,1]
  y = scaler.inverse_transform(y)
  y= y[:,0]

  # inverse log-transforming
  y = np.exp(y)[0]
  return y

def reshape_data_for_prediction(data_to_be_predicted):

    scalers = joblib.load('scalers.joblib')

    X = []

    for i in range(0,len(data_to_be_predicted)):

      df = data_to_be_predicted[i]
      scaler = scalers[i]

      # log transform
      price_col = [i for i in df.columns if "price" in i][0]
      df[price_col] = np.log(df[price_col])

      data = df.values

      X_arr = scaler.transform(data)

      X.append(X_arr)

    # stacking data three-dimensionally
    X = np.stack(X, axis= 2).reshape(len(X[0]),-1)

    X = np.expand_dims(X, axis= 0)

    return X

def get_prediction(data_to_be_predicted, model, coins= coins, horizon= horizon):

    predictions = model.predict(data_to_be_predicted)

    pred_df = pd.DataFrame(predictions.reshape(horizon, coins))

    return pred_df


def reshape_predicted_data(prediction_dataframe, list_of_dfs=list_of_dfs):

    scalers = joblib.load('scalers.joblib')

    for i in range(0,prediction_dataframe.shape[1]):

      scaler = scalers[i]
      prediction_dataframe[i] = prediction_dataframe[i].apply(lambda x: inverse_transformer(x,scaler))

    # renaming columns
    prediction_dataframe.columns = list_of_dfs

    prediction_dict = prediction_dataframe.to_dict(orient="list")

    return prediction_dict

# def preprocess_prediction(df):
#     """method that pre-process the data for prediction"""
#     # log transforming the data
#     df["high"] = np.log(df["high"])

#     # instantiating the scaler
#     scaler = joblib.load('crypto_prediction/scaler.joblib')

#     # selecting relevant column from df
#     dataset = df.values

#     # scaling the data
#     dataset_scaled = scaler.transform(dataset)

#     dataset_scaled = dataset_scaled.reshape(1,dataset_scaled.shape[0],dataset_scaled.shape[1])

#     return dataset_scaled

# def inverse_scale_prediction(pred):

#     scaler = joblib.load('crypto_prediction/scaler.joblib')

#     pred = inverse_transformer(pred, scaler)

#     pred = np.exp(pred)

#     return pred


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
