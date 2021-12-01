import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import Sequential, load_model, save_model, Model, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from crypto_prediction.utils import get_X_y, inverse_transformer
from google.cloud import storage

BUCKET_NAME = 'crypto_prediction'

#BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

MODEL_NAME = 'crypto_prediction'

MODEL_VERSION = 'v1'

STORAGE_LOCATION = 'models/'

list_of_dfs = ["ban", "cummies", "dinu", "doge",
"doggy", "elon", "erc20", "ftm", "grlc", "hoge",
"lowb", "mona", "samo", "shib", "shibx", "smi",
"wow", "yooshi","yummy"]

project_id = 'crypto-prediction-333213'

horizon = 24

history_size = 48

coins = 19

def get_data():

    dfs = []

    for df in list_of_dfs:

        sql = f"""
        SELECT *
        FROM `crypto-prediction-333213.crypto_BQ.{df}`;
        """

        dfs.append(pd.read_gbq(sql, project_id=project_id, dialect='standard'))

    return dfs

def data_cleaning(dfs):

    # takes as input the list of dataframes

    dfs_new = []

    for i in range(0,len(dfs)):
        df = dfs[i]
        price_col = [i for i in df.columns if "price" in i][0]

        # data cleaning
        df = df.sort_values(by="datetime")
        df = df.drop_duplicates(keep= "last")
        df = df.set_index("datetime")
        df[price_col].interpolate(method="linear", inplace= True)

        # adding financial indicators
        df["hourly_pct_change"] = df[price_col].pct_change(1)
        df["MA12_hours"] = df[price_col].rolling(window=12).mean()
        df["MA72_hours"] = df[price_col].rolling(window=72).mean()
        df["MAshort_over_long"] = df["MA12_hours"] > df["MA72_hours"]
        df["MAshort_over_long"] = df["MAshort_over_long"].apply(lambda x: 1 if x == True else 0)

        cols = [i for i in df.columns[2:]]
        for col in cols:
            df[col] = df[col].fillna(-99)

        dfs_new.append(df)

    return dfs_new


def reshape_data(dfs, history_size):

    # scaling the data & get_X_y for all datasets
    X = []
    y = []
    scalers = []

    for df in dfs:

        # log transform
        price_col = [i for i in df.columns if "price" in i][0]
        df[price_col] = np.log(df[price_col])

        data = df.values

        # scaling
        scaler = MinMaxScaler()

        scaler.fit(data)
        data = scaler.transform(data)

        # get X and y
        X_arr, y_arr = get_X_y(history_size, horizon, data)

        X.append(X_arr)
        y.append(y_arr)
        scalers.append(scaler)

    # stacking data three-dimensionally
    X = np.stack(X, axis= 2).reshape(len(X[0]),history_size,-1)
    y = np.stack(y, axis = 2).reshape(len(y[0]),-1)

    return X, y, scalers

def modeling(coins, horizon):

    # instantiating a model
    model = Sequential()

    # first network layer
    model.add(Masking(mask_value=-99))
    model.add(LSTM(units = 100, activation= "tanh", return_sequences= True))
    model.add(Dropout(0.2))

    # network layer's 2 - 5
    model.add(LSTM(units= 100, activation= "tanh", return_sequences= True))
    model.add(LSTM(units= 10, activation= "tanh", return_sequences= False))
    model.add(Dropout(0.2))

    # network output layer
    model.add(Dense(coins*horizon, activation= "linear"))

    model.compile(optimizer= "adam", loss= "mse")

    return model

def train_model(model, X, y):
    '''function that trains the model'''

    es = EarlyStopping(patience = 50, restore_best_weights= True)

    print(X.shape)
    print(y.shape)

    model.fit(X,
            y,
            validation_split= 0.2,
            epochs = 1,
            batch_size= 64,
            callbacks= [es],
            verbose= 1)

    return model

def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION + 'model_architecture.json')
    blob.upload_from_filename('model_architecture.json')

    blob = bucket.blob(STORAGE_LOCATION + 'model_weights.h5')
    blob.upload_from_filename('model_weights.h5')


def save_model(model):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    json_model = model.to_json()
    print("json to model worked")
    open('model_architecture.json', 'w').write(json_model)
    print("saved json model")
    # saving weights
    model.save_weights('model_weights.h5', overwrite=True)
    print("saved json model weights")



    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':

    dfs = get_data()

    dfs = data_cleaning(dfs)

    X, y, scalers = reshape_data(dfs, history_size)

    joblib.dump(scalers, open('scalers.joblib', 'wb'))
    print("scalers saved")


    model = modeling(coins, horizon)
    print("\nmodel compiled")

    model = train_model(model, X, y)
    print("\ntraining worked")

    save_model(model)
    print("\nmodel uploaded to GCP")
