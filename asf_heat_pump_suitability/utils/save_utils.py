import logging
import pickle

import polars as pl
from sklearn.base import BaseEstimator

from asf_heat_pump_suitability.utils.storage import get_boto3_client, get_s3fs


def save_model_to_pkl_s3(model: BaseEstimator, path: str) -> None:
    """Save a scikit-learn estimator as a pickle file to S3.

    Args:
        model: Trained scikit-learn estimator.
        path: S3 URI destination.
    """
    fs = get_s3fs()
    with fs.open(path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Saved model to {path}")


def save_to_s3(df: pl.DataFrame, path: str) -> None:
    """
    Save dataframe as parquet file to S3.

    Args:
        df (pl.DataFrame): dataframe
        path (str): path to S3 destination

    Returns:
        None
    """
    logging.info(f"Saving file to {path}")
    file_type = path.split(".")[-1]
    fs = get_s3fs()
    if file_type == "parquet":
        with fs.open(path=path, mode="wb") as f:
            df.write_parquet(f)
    elif file_type == "csv":
        with fs.open(path=path, mode="wb") as f:
            df.write_csv(f)
    else:
        raise ValueError(
            "Save to S3 can only save .parquet or .csv file types."
            "Please ensure the `path` argument contains one of these file types."
        )


def upload_file_to_s3(
    local_file_path: str,
    s3_bucket: str,
    s3_key_dir: str,
    filename: str,
    subfolder: str,
) -> None:
    """
    Upload a local file to an S3 bucket.

    Args:
        local_file_path (str): Path to the local file.
        s3_bucket (str): Name of the S3 bucket.
        s3_key_dir (str): S3 key (path) where the file should be uploaded.
        filename (str): The actual filename to store in S3.
        subfolder (str): Subfolder within S3.
    """
    s3_client = get_boto3_client("s3")
    s3_key = f"{s3_key_dir}{subfolder}/{filename}"
    s3_client.upload_file(local_file_path, s3_bucket, s3_key)
    logging.info(f"File uploaded to s3://{s3_bucket}/{s3_key}")
