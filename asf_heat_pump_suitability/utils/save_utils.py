import fsspec
import s3fs
import polars as pl
import geopandas as gpd
import logging
import boto3
from datetime import datetime
from sklearn.base import BaseEstimator
import pickle
import json

from asf_heat_pump_suitability import config


def get_str_release_date(release_date: str | None = None) -> str:
    """
    Validate a release date string or default to today's date.

    The date format is set by `config["constant"]["release_date_format"]` (currently
    `%Y%m%d`, i.e. YYYYMMDD).

    Args:
        release_date (str | None): release date in the configured format, or None to use today's date

    Returns:
        str: validated release date in the configured format

    Raises:
        ValueError: if `release_date` does not match the configured format
    """
    date_format = config["constant"]["release_date_format"]
    if release_date is None:
        return datetime.today().strftime(date_format)
    try:
        parsed_date = datetime.strptime(release_date, date_format)
        # strptime is lenient (e.g. "080726" parses as year 807); require an exact
        # round-trip so only strings exactly matching the configured format are accepted
        if parsed_date.strftime(date_format) != release_date:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"release_date must be a valid date in the configured format, "
            f"e.g. '{datetime.today().strftime(date_format)}', got '{release_date}'."
        )
    return release_date


def get_str_output_path(
    dataset: str,
    release_date: str,
    check_exists: bool = False,
    **format_kwargs,
) -> str:
    """
    Build the S3 path for an output dataset in its dated release directory.

    Args:
        dataset (str): key of the output dataset path template in `config["output"]["dataset"]`
        release_date (str): release date, resolved once per run via
            `get_str_release_date` at the script entrypoint
        check_exists (bool): if True, raise FileNotFoundError when no file exists at the path.
            Set when reading upstream pipeline outputs to fail fast on a missing release.
        **format_kwargs: values for the remaining placeholders in the path template, e.g.
            `local_authority` or `local_authorities`

    Returns:
        str: S3 path to the output dataset file

    Raises:
        FileNotFoundError: if `check_exists` is True and no file exists at the path
    """
    path = config["output"]["dataset"][dataset].format(
        release_date=get_str_release_date(release_date), **format_kwargs
    )
    if check_exists and not s3fs.S3FileSystem().exists(path):
        raise FileNotFoundError(
            f"No file found at {path}. Has the upstream pipeline stage been run for this "
            "release date, or did you mean to pass --release_date for an existing release?"
        )
    return path


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
                json.dump(df, f, ensure_ascii=False)
        else:
            raise ValueError(
                "Save to S3 can only save dict as .geojson file types."
                "Please ensure the `path` argument contains .geojson file type."
            )
    elif isinstance(df, list):
        if file_type == "json":
            with fsspec.open(path, "w") as f:
                json.dump(df, f, ensure_ascii=False)
        else:
            raise ValueError(
                "Save to S3 can only save list as .json file types."
                "Please ensure the `path` argument contains .json file type."
            )
    else:
        raise TypeError(
            f"Can only save polars.DataFrame, geopandas.GeoDataFrame, dict, or list, not {type(df)}"
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
