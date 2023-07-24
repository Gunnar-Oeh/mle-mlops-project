from fastapi import FastAPI
from data_model import TaxiRide, TaxiRidePrediction
from predict import predict
import requests
import json
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app) # exposes FastAPI to evidently


@app.get("/") # / as an endpoint relate to the front-page of the api/app
def index():
    return {"message": "NYC Green-Taxi Ride Duration Prediction"}

@app.post("/predict", response_model=TaxiRidePrediction)
def predict_duration(data: TaxiRide):
    '''
    Sends the known post-request to the FastAPI but also sends a defined post using request,post
    to the evidently service in the json representation of the TaxiRidePrediction model'''
    prediction = predict("green-taxi-ride-duration-3", data) # predict function from predict.py
    try:
        # Send post-request to another service: evidently
        response = requests.post(
            # where to post data= to (URL):
            f"http://evidently_service:8085/iterate/green_taxi_data",
            # In the form of the TaxiRidePredictionModel
            # data as the argument: what should be posted
            data=TaxiRidePrediction(
                **data.dict(), prediction=prediction
            # returned as a json-representation of the data-model 
            ).model_dump_json()
            ,
            # how to post it, in which format
            headers={"content-type": "application/json"},
        )
    except requests.exceptions.ConnectionError as error:
        print(f"Cannot reach a metrics application, error: {error}, data: {data}")

    return TaxiRidePrediction(**data.dict(), prediction=prediction)