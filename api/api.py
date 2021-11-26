from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from crypto_prediction.utils import preprocess_prediction, inverse_scale_prediction
from crypto_prediction.gcp import download_model

from crypto_prediction.data import prediction_ready_df

from crypto_prediction.utils import date2utc_ts, gecko_make_df

#from datetime import datetime
#import pytz
#import pandas as pd
#import joblib
#from google.cloud import storage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"checking": "basic api works"}

@app.get("/ping")
def pingpong():
    return 'pong'

@app.get("/get_coin_history")
def get_coin_history(coin, start_date, end_date, interval='1d'):
    return('not active')

@app.get("/predict")
def get_prediction(coin_name):

    # here the api-calls have to be made to get historical price data
    # and google_trends data for the past 2 days, stored as a dataframe

    df = prediction_ready_df(coin_name)

    model = download_model()

    df_pred = preprocess_prediction(df)

    pred = model.predict(df_pred)

    prediction = inverse_scale_prediction(pred)

    return {'prediction':prediction[0]}



if __name__ == '__main__':
    df = prediction_ready_df('doge')
    print(df)
    pass
