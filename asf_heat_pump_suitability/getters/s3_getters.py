"""
Data getters (and savers)
"""

from fnmatch import fnmatch
import json
import pickle
import gzip
import os

import pandas as pd
import boto3
from decimal import Decimal
import numpy
import yaml
import io
from io import BytesIO
import geopandas as gpd

from asf_heat_pump_suitability import logger, PROJECT_DIR
from typing import List, Any, NoReturn
import requests


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super(CustomJsonEncoder, self).default(obj)


def get_s3_resource():
    s3 = boto3.resource("s3")
    return s3


def save_to_s3(bucket_name: str, output_var: Any, output_file_path: str) -> NoReturn:
    """
    Save a variable (of any type) to S3

    Args:
        bucket_name (str): Name of the S3 bucket you are saving to
        output_var (Any): The variable you want to save
        output_file_path (str): Name of the output path

    Returns:
        Nothing
    """
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, output_file_path)

    if fnmatch(output_file_path, "*.csv"):
        output_var.to_csv("s3://" + bucket_name + "/" + output_file_path, index=False)
    elif fnmatch(output_file_path, "*.parquet"):
        output_var.to_parquet(
            "s3://" + bucket_name + "/" + output_file_path, index=False
        )
    elif fnmatch(output_file_path, "*.pkl") or fnmatch(output_file_path, "*.pickle"):
        obj.put(Body=pickle.dumps(output_var))
    elif fnmatch(output_file_path, "*.gz"):
        obj.put(Body=gzip.compress(json.dumps(output_var).encode()))
    elif fnmatch(output_file_path, "*.txt"):
        obj.put(Body=output_var)
    elif (
        fnmatch(output_file_path, "*.jpg")
        or fnmatch(output_file_path, "*.png")
        or fnmatch(output_file_path, "*.jpeg")
    ):
        image_data = BytesIO(output_var)
        obj.put(Body=image_data)
    elif fnmatch(output_file_path, "*.json"):
        obj.put(Body=json.dumps(output_var, cls=CustomJsonEncoder))
    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.parquet", "*.jsonl.gz", "*.jsonl", "*.json", "*.png", "*.jpeg".'
        )


def load_s3_data(
    bucket_name: str,
    file_name: str,
) -> Any:
    """
    Load data from S3 location.

    Args:
        bucket_name (str): Name of the S3 bucket you are saving to
        file_name (str): S3 key to load

    Returns:
        (Any): The file you have loaded
    """
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return [json.loads(line) for line in file]
    if fnmatch(file_name, "*.yml") or fnmatch(file_name, "*.yaml"):
        file = obj.get()["Body"].read().decode()
        return yaml.safe_load(file)
    elif fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.json.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return json.load(file)
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_name, "*.csv"):
        return pd.read_csv("s3://" + bucket_name + "/" + file_name)
    elif fnmatch(file_name, "*.parquet"):
        return pd.read_parquet("s3://" + bucket_name + "/" + file_name)
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    elif fnmatch(file_name, "*.gpkg"):
        with BytesIO(obj.get()["Body"].read()) as file:
            return gpd.read_file(file)
    elif fnmatch(file_name, "*.geojson"):
        with BytesIO(obj.get()["Body"].read()) as file:
            return gpd.read_file(file)
    elif (
        fnmatch(file_name, "*.jpg")
        or fnmatch(file_name, "*.png")
        or fnmatch(file_name, "*.jpeg")
    ):
        # Download the image from S3 into a BytesIO object
        image_data = BytesIO()
        obj.download_fileobj(image_data)
        return image_data

    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.parquet", "*.gpkg", "*.geojson", "*.jsonl.gz", "*.jsonl", or "*.json"'
        )


def dictionary_to_s3(
    data_dict: dict, s3_bucket: str, s3_folder: str, file_name: str
) -> NoReturn:
    """
    Transforms a dictionary into a json and uploads to S3.

    Args:
        data_dict (dict): Dictionary to be saved.
        s3_bucket (str): S3 bucket name where to upload the file
        s3_folder (str): folder where to store the file within the S3 bucket
        file_name (str): name of the file

    Returns:
        Nothing
    """
    s3_client = boto3.client("s3")
    obj = io.BytesIO(json.dumps(data_dict).encode("utf-8"))
    s3_client.upload_fileobj(obj, s3_bucket, os.path.join(s3_folder, file_name))


def read_json_from_s3(bucket: str, file_path: str) -> dict:
    """
    Reads a json file from S3 without downloading it.

    Args:
        bucket (str): S3 bucket name
        file_path (str): file path (including file name)

    Returns:
        (dict): The file you have loaded
    """
    s3_resource = boto3.resource("s3")
    json_file = s3_resource.Object(bucket, file_path)
    json_file = json_file.get()["Body"].read().decode("utf-8")
    return json.loads(json_file)
