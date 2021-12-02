from google.cloud import storage
import pandas as pd
#import pandas_gbq
from tensorflow.keras.models import model_from_json
from datetime import datetime, timedelta

BUCKET_NAME = "crypto_prediction"

list_of_dfs = ["ban", "cummies", "dinu", "doge",
"doggy", "elon", "erc20", "ftm", "grlc", "hoge",
"lowb", "mona", "samo", "shib", "shibx", "smi",
"wow", "yooshi","yummy"]

project_id = 'crypto-prediction-333213'


def download_model(bucket=BUCKET_NAME):

    client = storage.Client().bucket(bucket)

    storage_location_arch = 'models/model_architecture.json'
    storage_location_weights = 'models/model_weights.h5'

    architecture = client.blob(storage_location_arch)
    weights = client.blob(storage_location_weights)

    architecture.download_to_filename('model_architecture.json')
    weights.download_to_filename('model_weights.h5')
    model = model_from_json(open('model_architecture.json').read())

    model.load_weights('model_weights.h5')

    return model

def download_prediction_data(list_of_dfs=list_of_dfs):

    dfs = []

    for df in list_of_dfs:

        sql = f"""
        SELECT *
        FROM `crypto-prediction-333213.crypto_BQDB.{df}`
        ORDER BY datetime DESC LIMIT 100;
        """

        dfs.append(pd.read_gbq(sql, project_id=project_id, dialect='standard'))

    return dfs


if __name__ == "__main__":

    model = download_model()

    print(model)
