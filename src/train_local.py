### Here the script is not an interactive service.
### It's only purpose is to train the model again, possibly with new data, if being called
import os
import argparse
import pandas as pd
import mlflow
from mlflow.tracking.client import MlflowClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

features = ["PULocationID", "DOLocationID", "trip_distance", "passenger_count", 
                "fare_amount", "total_amount"]
target = 'duration'

# Download Data - needs to be run in working-directory as src/train.py
def download_data(color, year, month):
    # Check if data is already downloaded
    if not os.path.exists(f"./data/{color}_tripdata_{year}-{month:02d}.parquet"):
        print(f"Downloading {color}_tripdata_{year}-{month:02d}.parquet")
        os.system(f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet")
        print("Download finished")
    # Load the data
    print(f"Reading {color}_tripdata_{year}-{month:02d}.parquet")
    df = pd.read_parquet(f"./data/{color}_tripdata_{year}-{month:02d}.parquet")
    print("Import finished")
    return df

# Helper function to calculate trip duration
def calculate_trip_duration_in_minutes(df, features = features, target = target):
    df["duration"] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 8)]
    features.append(target)
    df = df[features]
    return df

# Preprocessing steps including trip_duration
def preprocess(df, features = features, target = target):
    df = df.copy()
    df_processed = calculate_trip_duration_in_minutes(df)

    y = df_processed[target]
    X = df_processed.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    return X_train, X_test, y_train, y_test

# Model Training
def train_model(X_train, X_test, y_train, y_test, color, year, month, features = features, target = target):
    
    # load and set environment variables
    load_dotenv()

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    SA_KEY= os.getenv("SA_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    print("Starting ML-Flow-Run")
    with mlflow.start_run():
    
        tags = {
            "model": "linear regression",
            "developer": "Gunnar",
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": features,
            "target": target
        }
        mlflow.set_tags(tags)
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred_train = lr.predict(X_train)
        y_pred = lr.predict(X_test)
        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        rmse_test = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse_test)
    
        mlflow.sklearn.log_model(lr, "model")
        run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/model"
        model_name = f"{color}-taxi-ride-duration-3"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        model_version = 1
        new_stage = "Production"
        client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=False
    )
    if cml_run: # Logs the error of the model
        with open("metrics.txt", "w") as f:
            f.write(f"RMSE on the Train Set: {rmse_train}")
            f.write(f"RMSE on the Test Set: {rmse_test}")

    print("model-training finished")

def main_train_model(color, year, month, cml_run):

    # Download data
    df_taxi = download_data(color, year, month)

    # preprocess data
    X_train, X_test, y_train, y_test = preprocess(df_taxi)
    train_model(X_train, X_test, y_train, y_test, color, year, month)

if __name__ == "__main__":
    # Get command-line-arguments
    parser = argparse.ArgumentParser()  # define command-line interface
    # Implement Continuous ML as CI/CD for this project
    parser.add_argument(                # 
        "--cml_run", default=False, action=argparse.BooleanOptionalAction, # Check for boolean Argument
        required=True)
    parser.add_argument("--color", type=str)
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    args = parser.parse_args()          # supplied arguments are stored

    # Store the attributes as global variables
    for attr in vars(args):
        globals()[attr] = getattr(args, attr) # assingns the value of args.attr of name attr
                                              # to a global variable of the same name
    
    main_train_model(color, year, month, cml_run)