from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from yahoo_fin.stock_info import get_data

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

@app.get("/get_coin_history")
def predict(coinlist):
    #sanizize and stuff


    return {"coinlist": coinlist}


if __name__ == '__main__':
    pass
