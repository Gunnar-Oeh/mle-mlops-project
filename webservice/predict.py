import mlflow
import os
from dotenv import load_dotenv
import pandas as pd


def load_model(model_name):
    stage = "Production"
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model_name, data):
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    SA_KEY = "./comp_eng_def_serv_acc.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Data input:", data)
    model_input = pd.DataFrame([data.dict()])
    print("Load model...")
    model = load_model(model_name)
    print("Making prediction with data: ", model_input.head())
    prediction = model.predict(model_input)
    return float(prediction[0])