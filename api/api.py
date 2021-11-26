from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from yahoo_fin.stock_info import get_data
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
    #return {"checking": "basic api works"}
    return {
        'usage':
        '',
        ' url_base':
        '/get_coin_history?',
        ' variables':
        'coin=doge-eur&start_date=23/11/2019&end_date=23/11/2021',
        ' optional':
        '&interval=1d (is default)',
        ' ':
        '',
        ' -- DANGER --':
        'data is not fully cleaned, might contain NANs or other artifacts',
        ' full url':
        '/get_coin_history?coin=doge-eur&start_date=23/11/2019&end_date=23/11/2021'
    }

@app.get("/get_coin_history")
def get_coin_history(coin, start_date, end_date, interval='1d'):
    #sanizize and stuff

    # df = get_data("doge-eur", start_date="23/11/2019", end_date="23/11/2021", index_as_date = True, interval="1d")
    df = get_data(coin, start_date=start_date, end_date=end_date, index_as_date = True, interval=interval)
    df_json = df.to_json()

    return(df_json)

@app.get("/predict")
def get_prediction(coin_name):

    array = # implement function to transform inputs into shape that LSTM Model can make a prediction on

    model = download_model()

    prediction = model.predict(array)

    return prediction



if __name__ == '__main__':
    pass
