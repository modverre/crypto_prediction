import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model, save_model, Model, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from utils import get_X_y, inverse_transformer
from google.cloud import storage

BUCKET_NAME = 'crypto_prediction'

#BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

MODEL_NAME = 'crypto_prediction'

MODEL_VERSION = 'v1'

STORAGE_LOCATION = 'models/'


def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""

    # getting historical price data
    df = pd.read_csv('./data/doge-hist-2y.csv')
    df.rename(columns= {"Unnamed: 0": "Date"}, inplace= True)
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format= True)
    df = df.set_index("Date")
    df.interpolate(method= "linear", inplace= True)

    # getting daily google trends data
    df2 = pd.read_csv('./data/doge_daily_google_trends_1y.csv')
    df2["date"] = pd.to_datetime(df2["date"], infer_datetime_format= True)
    df2 = df2.set_index("date")
    df2.interpolate(method= "linear", inplace= True)

    # joining both dataframes
    df_final = df[["high"]].join(df2[["Dogecoin"]], how= "outer")
    df_final.rename(columns={"Dogecoin": "Google_Trends"}, inplace= True)
    df_final.dropna(inplace= True)

    return df_final


def preprocess(df):
    """method that pre-process the data"""

    # log transforming the data
    df["high"] = np.log(df["high"])

    # instantiating the scaler
    scaler = MinMaxScaler()

    # selecting relevant column from df
    dataset = df.values

    # scaling the data
    dataset_scaled = scaler.fit_transform(dataset)

    # splitting into train and test data
    split = int(dataset.shape[0]*0.8)
    train, test = dataset[:split], dataset[split:]

    # selecting nr. of days used to predict next value
    history_size = 2

    # creating arrays X, y for train and test data
    X_train, y_train = get_X_y(history_size, train)

    X_test, y_test = get_X_y(history_size, test)

    # reshaping X_train and X_test
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],2))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],2))

    return X_train, X_test, y_train, y_test, scaler

def compile_model(X_train, history_size= 2):
    '''function that instantiates and compiles the model'''

    # instantiating a model
    model = Sequential()

    # first network layer
    model.add(LSTM(units = 100, return_sequences= True, input_shape = (history_size, 2)))
    model.add(Dropout(0.2))

    # network layer's 2 - 5
    model.add(LSTM(units= 50, return_sequences= True))
    model.add(Dropout(0.2))
    model.add(LSTM(units= 10, return_sequences= False))
    model.add(Dropout(0.2))

    # network output layer
    model.add(Dense(units= 1))

    model.compile(optimizer= "adam", loss= "mse")

    return model


def train_model(model, X_train, y_train):
    '''function that trains the model'''

    es = EarlyStopping(patience = 100, restore_best_weights= True)

    model.fit(X_train,
            y_train,
            validation_split= 0.2,
            epochs = 1000,
            batch_size= 32,
            callbacks= [es],
            verbose= 1)

    return model

def evaluate_model(model, scaler, X_test, y_test):
    '''function that evalutes the model performance'''

    # inverse transforming the data
    real_stock_price = inverse_transformer(y_test, scaler)
    predicted_stock_price = inverse_transformer(model.predict(X_test), scaler)

    # inverse log transforming the date
    real_stock_price = np.exp(real_stock_price)
    predicted_stock_price = np.exp(predicted_stock_price)

    # evaluating model performance
    rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

    print(f"RMSE = {rmse}")


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

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    # print("test1")
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    # print("test2")

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

    df = get_data()
    print("\ndata received")

    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print("\ndata preprocessed")

    # make_keras_picklable()

    model = compile_model(X_train)
    print("\nmodel compiled")

    model = train_model(model, X_train, y_train)
    print("\ntraining worked")

    evaluate_model(model, scaler, X_test, y_test)
    print("\nmodel training and evaluation complete")

    #save_model(model)
    #print("\nmodel uploaded to GCP")
