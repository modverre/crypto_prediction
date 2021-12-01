from inspect import _ParameterKind
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd


from crypto_prediction.utils import reshape_data_for_prediction, reshape_predicted_data, get_prediction
from crypto_prediction.gcp import download_model, download_prediction_data
from crypto_prediction.data import prediction_ready_df, coin_history
from crypto_prediction.model import data_cleaning


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

@app.get("/get/coin_history")
def get_coin_history(tickerlist, hoursback):
    """
    input:
        tickerlist      - ticker names seperated by comma: samo,doge,shib ..
        hoursback       - how many hours to look back (could take dates, too, not yet connected)

    output:
        dict
    """
    # we should sanitize here since its unknown input
    # ...

    tickerlist = tickerlist.split(',')

    return coin_history(tickerlist, int(hoursback))

@app.get("/predict")
def predict_endpoint():

    model = download_model()
    print("downloaded model")

    data =  download_prediction_data()
    print("downloaded data from GBQ")

    data = data_cleaning(data)
    print("data cleaned")

    data =  reshape_data_for_prediction(data)
    print("data reshaped")

    predictions = get_prediction(data, model)
    print("model predicted data")

    output = reshape_predicted_data(predictions)
    print("output ready")

    return output

if __name__ == '__main__':

    output = predict_endpoint()

    print(output)
