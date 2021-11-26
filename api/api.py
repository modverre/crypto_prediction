from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from crypto_prediction.gcp import download_model

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
    #return {
    #    'usage':
    #    '',
    #    ' url_base':
    #    '/get_coin_history?',
    #    ' variables':
    #    'coin=doge-eur&start_date=23/11/2019&end_date=23/11/2021',
    #    ' optional':
    #    '&interval=1d (is default)',
    #    ' ':
    #    '',
    #    ' -- DANGER --':
    #    'data is not fully cleaned, might contain NANs or other artifacts',
    #    ' full url':
    #    '/get_coin_history?coin=doge-eur&start_date=23/11/2019&end_date=23/11/2021'
    #}

@app.get("/ping")
def pingpong():
    return 'pong'


@app.get("/get_coin_history")
def get_coin_history(coin, start_date, end_date, interval='1d'):
    return('not active')

@app.get("/predict")
def get_prediction(price_hist, google_trends):

    array = [] # implement function to transform inputs into shape that LSTM Model can make a prediction on

    model = download_model()

    prediction = model.predict(array)

    return prediction



if __name__ == '__main__':
    pass
