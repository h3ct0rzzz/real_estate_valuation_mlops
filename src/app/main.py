from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
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


ERROR_MESSAGES = {
    "building_type": "Для деревянного здания количество этажей не может быть больше 4",
    "levels_comparison": "Количество этажей не может быть меньше текущего этажа",
    "area_comparison": "Общая площадь не может быть меньше площади кухни",
    "level_threshold": "Этаж не может быть больше 60"
}


load_dotenv(override=True)

S3_BUCKET = str(os.getenv('S3_BUCKET'))
S3_DATASETS_DATASET = str(os.getenv('S3_DATASETS_DATASET'))
S3_DATASETS_GEO = str(os.getenv('S3_DATASETS_GEO'))
S3_DATASETS_STATIONS = str(os.getenv('S3_DATASETS_STATIONS'))

AWS_ACCESS_KEY_ID = str(os.getenv('AWS_ACCESS_KEY_ID'))
AWS_SECRET_ACCESS_KEY = str(os.getenv('AWS_SECRET_ACCESS_KEY'))
AWS_REGION = str(os.getenv('AWS_REGION'))

MLFLOW_TRACKING_URI = str(os.getenv('MLFLOW_TRACKING_URI'))
MLFLOW_S3_ENDPOINT_URL = str(os.getenv('MLFLOW_S3_ENDPOINT_URL'))
RUN_ID = str(os.getenv('RUN_ID'))


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3: boto3.client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                endpoint_url=MLFLOW_S3_ENDPOINT_URL)

mlflow_client: MlflowClient = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

datasets: Dict[str, pd.DataFrame] = {}
model: mlflow.pyfunc.PyFuncModel = None


def get_s3_object(s3_client: boto3.client, key: str) -> Dict[str, Any]:
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    return obj


def read_dataset_from_s3(obj):
    buffer = io.BytesIO(obj['Body'].read())
    return pd.read_parquet(buffer)


def load_datasets() -> Dict[str, pd.DataFrame]:
    if not datasets:
        obj_geo = get_s3_object(s3, S3_DATASETS_GEO)
        obj_stations = get_s3_object(s3, S3_DATASETS_STATIONS)
        # obj_datasets = get_s3_object(s3, S3_DATASETS_DATASET)
        datasets['geo'] = read_dataset_from_s3(obj_geo)
        datasets['stations'] = read_dataset_from_s3(obj_stations)
        # datasets['dataset'] = read_dataset_from_s3(obj_datasets)
        # datasets['dataset'] = reduce_mem_usage(datasets['dataset'])
    return datasets


def load_model_from_mlflow() -> mlflow.pyfunc.PyFuncModel:
    global model
    if not model:
        logged_model = RUN_ID
        model = mlflow.pyfunc.load_model(logged_model)
    return model


@app.post("/predict")
async def predict_endpoint(request: Request) -> Response:
    #try:
    json_data: dict = await request.json()
    if not json_data:
        return Response(status_code=400,
                        content=json.dumps({"data": None,
                                            "error": "Invalid request body",
                                            "status": 400}),
                        media_type="application/json")

    df: pd.DataFrame = json_to_dataframe(json_data)
    if df.empty:
        return Response(status_code=400,
                        content=json.dumps({"data": None,
                                            "error": "Invalid JSON data",
                                            "status": 400}),
                        media_type="application/json")

    data = df.to_dict('records')[0]

    error_map = {
        "building_type": lambda x: x == "Деревянный" and data["levels"] > 4,
        "levels_comparison": lambda x: data["levels"] < data["level"],
        "area_comparison": lambda x: data["area"] < data["kitchen_area"],
        "level_threshold": lambda x: data["level"] > 60
    }

    for field, check in error_map.items():
        if check(field):
            error_message = ERROR_MESSAGES[field]
            return Response(status_code=400,
                            content=json.dumps({"data": None,
                                                "error": error_message,
                                                "status": 400}),
                            media_type="application/json")

    datasets: Dict[str, pd.DataFrame] = load_datasets()
    geo: pd.DataFrame = datasets['geo']
    stations: pd.DataFrame = datasets['stations']

    model: mlflow.pyfunc.PyFuncModel = load_model_from_mlflow()

    df: pd.DataFrame = add_features(df, geo, stations)
    df['area'] = df['area'].astype(np.float32)
    df['kitchen_area'] = df['area'].astype(np.float32)
    predict = model.predict(df.drop(["street", "house_number", "geo_lat", "geo_lon"], axis=1))
     df["price"] = np.expm1(predict) * df["area"]

    return Response(status_code=200, content=json.dumps(
        {"data": df.to_dict(orient='records'), "error": None, "status": 200}), media_type="application/json")
    '''
    except Exception as e:
        return Response(status_code=500, content=json.dumps(
            {"data": None, "error": str(e), "status": 500}), media_type="application/json")
    '''

async def run_server():
    u_config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8088,
        log_level="info",
        reload=True)
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
