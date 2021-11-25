from google.cloud import storage
from tensorflow.keras.models import model_from_json

BUCKET_NAME = "crypto_prediction"


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

if __name__ == "__main__":

    model = download_model()

    print(model)
