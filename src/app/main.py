from fastapi import FastAPI, Request, Response
import json
import pandas as pd
import os
import boto3
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from typing import Dict, Any
import uvicorn
import asyncio
import sys
from dotenv import load_dotenv
from src.data.make_dataset import add_features, json_to_dataframe
import io

load_dotenv(override=True)

S3_BUCKET = os.getenv('S3_BUCKET')
S3_DATASETS_DATASET = os.getenv('S3_DATASETS_DATASET')
S3_DATASETS_GEO = os.getenv('S3_DATASETS_GEO')
S3_DATASETS_STATIONS = os.getenv('S3_DATASETS_STATIONS')

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL')
RUN_ID = os.getenv('RUN_ID')

app = FastAPI()

s3: boto3.client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                endpoint_url=MLFLOW_S3_ENDPOINT_URL)

mlflow_client: MlflowClient = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

datasets: Dict[str, pd.DataFrame] = {}
model: mlflow.pyfunc.PyFuncModel = None


def get_s3_object(s3: boto3.client, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj


def read_dataset_from_s3(obj):
    buffer = io.BytesIO(obj['Body'].read())
    return pd.read_parquet(buffer)


def load_datasets() -> Dict[str, pd.DataFrame]:
    if not datasets:
        obj_geo = get_s3_object(s3, S3_DATASETS_GEO)
        obj_stations = get_s3_object(s3, S3_DATASETS_STATIONS)
        datasets['geo'] = read_dataset_from_s3(obj_geo)
        datasets['stations'] = read_dataset_from_s3(obj_stations)
    return datasets


def load_model_from_mlflow() -> mlflow.pyfunc.PyFuncModel:
    global model
    if not model:
        logged_model = RUN_ID
        model = mlflow.pyfunc.load_model(logged_model)
    return model


@app.post("/predict")
async def predict_endpoint(request: Request) -> Response:
    # Get JSON data from request
    json_data: str = await request.json()
    df: pd.DataFrame = json_to_dataframe(json_data)

    datasets: Dict[str, pd.DataFrame] = load_datasets()
    geo: pd.DataFrame = datasets['geo']
    stations: pd.DataFrame = datasets['stations']

    model: mlflow.pyfunc.PyFuncModel = load_model_from_mlflow()

    df: pd.DataFrame = add_features(df, geo, stations)

    predict = model.predict(df.drop(["street", "house_number"], axis=1))
    df["price"] = np.expm1(predict) * df["area"]

    return Response(content=json.dumps(df.to_dict()), media_type="application/json")


async def run_server():
    u_config = uvicorn.Config("main:app", host="0.0.0.0", port=8088, log_level="info", reload=True)
    server = uvicorn.Server(u_config)
    await server.serve()


async def main():
    tasks = [
        run_server(),
    ]

    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':
    if AWS_ACCESS_KEY_ID is None or AWS_SECRET_ACCESS_KEY is None:
        sys.exit(1)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
        loop.run_forever()
        loop.close()
    except KeyboardInterrupt:
        sys.exit(0)
