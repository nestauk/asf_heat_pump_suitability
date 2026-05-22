import fsspec
import s3fs
import polars as pl
import geopandas as gpd
import logging
import boto3
from sklearn.base import BaseEstimator
import pickle


def save_model_to_pkl_s3(model: BaseEstimator, path: str) -> None:
    """
    Save Estimator as pickle file to S3.

    Args:
        model (BaseEstimator): trained model
        path (str): path to S3 destination

    Returns:
        None
    """
    fs = s3fs.S3FileSystem()
    pickle.dump(model, fs.open(path, "wb"))
    print(f"Saved model to {path}")


def save_to_s3(df: pl.DataFrame | gpd.GeoDataFrame, path: str) -> None:
    """
    Save polars.DataFrame as parquet or csv file to S3, or save geopandas.GeoDataFrame as geoparquet, geojson, or other
    specified file type to S3.

    Args:
        df (pl.DataFrame | gpd.GeoDataFrame): dataframe or geodataframe
        path (str): path to S3 destination

    Returns:
        None

    Raises:
        ValueError: if file type not allowed
        TypeError: if `df` argument is not a polars.DataFrame or a geopandas.GeoDataFrame
    """
    logging.info(f"Saving file to {path}")
    file_type = path.split(".")[-1]
    fs = s3fs.S3FileSystem()
    if isinstance(df, pl.DataFrame):
        if file_type == "parquet":
            with fs.open(path=path, mode="wb") as f:
                df.write_parquet(f)
        elif file_type == "csv":
            with fs.open(path=path, mode="wb") as f:
                df.write_csv(f)
        else:
            raise ValueError(
                "Save to S3 can only save polars DataFrames .parquet or .csv file types."
                "Please ensure the `path` argument contains one of these file types."
            )
    elif isinstance(df, gpd.GeoDataFrame):
        if file_type == "parquet":
            df.to_parquet(path)
        elif file_type == "geojson":
            print("Converting CRS to EPSG:4326 and saving file as geojson...")
            df = df.to_crs(epsg=4326)
            with fsspec.open(path, "w") as f:
                f.write(df.to_json(drop_id=True))
        else:
            with fsspec.open(path, "wb") as f:
                df.to_file(f)
    elif isinstance(df, dict):
        if file_type == "geojson":
            with fsspec.open(path, "w") as f:
                import json

                json.dump(df, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError(
                "Save to S3 can only save dict GeoDataFrames as .geojson file types."
                "Please ensure the `path` argument contains .geojson file type."
            )
    else:
        raise TypeError(
            f"Can only save polars.DataFrame, geopandas.GeoDataFrame, or dict, not {type(df)}"
        )


def upload_file_to_s3(
    local_file_path: str,
    s3_bucket: str,
    s3_key_dir: str,
    filename: str,
    subfolder: str,
):
    """
    Upload a local file to an S3 bucket.

    Args:
        local_file_path (str): Path to the local file.
        s3_bucket (str): Name of the S3 bucket.
        s3_key_dir (str): S3 key (path) where the file should be uploaded.
        filename (str): The actual filename to store in S3.
        subfolder (str): Subfolder within S3.
    """
    s3_client = boto3.client("s3")
    s3_key = f"{s3_key_dir}{subfolder}/{filename}"
    s3_client.upload_file(local_file_path, s3_bucket, s3_key)
    logging.info(f"File uploaded to s3://{s3_bucket}/{s3_key}")
