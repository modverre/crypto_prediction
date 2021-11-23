#from google.cloud import storage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'crypto-prediction'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/doge-hist-2y.csv'


# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'crypto-prediction'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'



def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=1000)
    df = pd.read_csv('./data/doge-hist-2y.csv')
    return df


def preprocess(df):
    """method that pre-process the data"""

    df.rename(columns= {"Unnamed: 0": "Date"}, inplace= True)
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format= True)
    df = df.set_index("Date")
    df.interpolate(method= "linear", inplace= True)

    scaler = MinMaxScaler()

    # selecting relevant column from df
    dataset = df.iloc[:,1:2].values

    # scaling the data
    dataset_scaled = scaler.fit_transform(dataset)

    # splitting into train and test data
    train, test = train_test_split(dataset_scaled, shuffle = False)

    # selecting nr. of days used to predict next value
    history_size = 14

    # creating arrays X, y for train and test data
    X_train = []
    y_train = []

    for i in range(history_size, train.size):
        X_train.append(train[i-history_size:i,0])
        y_train.append(train[i,0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = []
    y_test = []

    for i in range(history_size, test.size):
        X_test.append(test[i-history_size:i,0])
        y_test.append(test[i,0])

    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

    return X_train, X_test, y_train, y_test

def compile_model(X_train):

    # instantiating a model
    model = Sequential()

    # first network layer
    model.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # network layer's 2 - 5
    model.add(LSTM(units= 50, return_sequences= True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences= True))
    model.add(Dropout(0.2))
    model.add(LSTM(units= 50, return_sequences= True))
    model.add(Dropout(0.2))
    model.add(LSTM(units= 50, return_sequences= True))
    model.add(Dropout(0.2))
    model.add(LSTM(units= 20, return_sequences= False))
    model.add(Dropout(0.2))

    # network output layer
    model.add(Dense(units= 1))

    model.compile(optimizer= "rmsprop", loss= "mse")

    return model


def train_model(model, X_train, y_train):
    """method that trains the model"""

    es = EarlyStopping(patience = 20, restore_best_weights= True)

    model.fit(X_train,
            y_train,
            validation_split= 0.2,
            epochs = 150,
            batch_size= 16,
            callbacks= [es],
            verbose= 1)

    print("trained model")
    return model

def evaluate_model(model, X_test, y_test):

    score = model.evaluate(X_test, y_test, verbose= 0)

    print(f"RMSE = {np.sqrt(score)}")


# STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'


# def upload_model_to_gcp():


#     client = storage.Client()

#     bucket = client.bucket(BUCKET_NAME)

#     blob = bucket.blob(STORAGE_LOCATION)

#     blob.upload_from_filename('model.joblib')


# def save_model(reg):
#     """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
#     HINTS : use joblib library and google-cloud-storage"""

#     # saving the trained model to disk is mandatory to then beeing able to upload it to storage
#     # Implement here
#     joblib.dump(reg, 'model.joblib')
#     print("saved model.joblib locally")

#     # Implement here
#     upload_model_to_gcp()
#     print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    # get training data from GCP bucket

    df = get_data()
    print("data received")

    X_train, X_test, y_train, y_test = preprocess(df)
    print("data preprocessed")

    model = compile_model(X_train)
    print("model compiled")
    print(X_train.shape[1])

    model = train_model(model, X_train, y_train)
    print("evaluation worked")

    evaluate_model(model, X_test, y_test)

    print("model training and evaluation complete")
