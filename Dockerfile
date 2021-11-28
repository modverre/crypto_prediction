FROM python:3.8.6-buster

COPY api /api
COPY crypto_prediction /crypto_prediction
COPY crypto_prediction/scaler.joblib /crypto_prediction/scaler.joblib
COPY requirements.txt /requirements.txt

COPY /crypto-prediction-333213-e8032c892bb2.json /credentials.json

RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
