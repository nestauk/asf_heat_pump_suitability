"""
Generic loaders of specific file types. These functions shouldn’t load specific datasets and can be used in multiple specific getters.
"""

import logging
import pickle
from fnmatch import fnmatch
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import geopandas as gpd
import polars as pl
import requests
import s3fs


def load_df(path: str, **kwargs) -> pl.DataFrame:
    """Load a Polars DataFrame from a local path or S3 URI.

    Dispatches to :func:`load_df_from_s3` for ``s3://`` paths so that
    credentials are resolved via s3fs rather than Polars' built-in S3 reader.
    Supports .parquet and .csv.

    Args:
        path (str): Local filesystem path or ``s3://`` URI.
        **kwargs: Forwarded to the underlying Polars reader.

    Returns:
        pl.DataFrame
    """
    if path.startswith("s3://"):
        return load_df_from_s3(path, **kwargs)
    if fnmatch(path, "*.parquet"):
        return pl.read_parquet(path, **kwargs)
    elif fnmatch(path, "*.csv"):
        return pl.read_csv(path, **kwargs)


def load_df_from_s3(uri: str, **kwargs) -> pl.DataFrame:
    """
    Load polars dataframe from S3.

    Uses s3fs for credential resolution so the local AWS credential chain
    (profile, env vars, etc.) is respected rather than Polars' built-in S3
    reader which defaults to EC2 instance metadata.

    Args:
        uri (str): S3 URI
        **kwargs for polars file reader.

    Returns:
        pl.DataFrame
    """
    fs = s3fs.S3FileSystem()
    with fs.open(uri, "rb") as f:
        if fnmatch(uri, "*.parquet"):
            return pl.read_parquet(f, **kwargs)
        elif fnmatch(uri, "*.csv"):
            return pl.read_csv(f, **kwargs)


def get_df_from_zip_csv_s3(path: str, extract_file: str, **kwargs) -> pl.DataFrame:
    """
    Load dataframe from csv in ZIP file stored an S3.

    Args:
        path (str): S3 URI of ZIP file load
        extract_file (str): name of file to extract
        **kwargs for pl.read_csv()

    Returns:
        pl.DataFrame: dataset from ZIP file
    """
    print(f"Loading file from path: {path}")
    content = BytesIO(get_content_from_s3_path(path))
    df = pl.read_csv(ZipFile(content).open(name=extract_file), **kwargs)

    return df


def get_df_from_excel_s3_path(path: str, **kwargs) -> pl.DataFrame:
    """
    Get dataframe from Excel file stored in s3 path.

    Args
        path (str): S3 URI to Excel file
        **kwargs for pl.read_excel()
    Returns
        pl.DataFrame: dataframe from Excel file
    """
    content = BytesIO(get_content_from_s3_path(path))
    df = pl.read_excel(content, **kwargs)
    return df


def get_df_from_csv_s3_path(path: str, **kwargs) -> pl.DataFrame:
    """
    Get dataframe from CSV file stored in s3 path.

    Args
        path (str): S3 URI to CSV file
        **kwargs for pl.read_csv()
    Returns
        pl.DataFrame: dataframe from CSV file
    """
    content = BytesIO(get_content_from_s3_path(path))
    df = pl.read_csv(content, **kwargs)
    return df


def get_content_from_s3_path(path: str) -> bytes:
    """
    Get bytes content of file from S3 path.

    Args
        path (str): S3 URI to file

    Returns
        bytes: bytes content of file
    """
    fs = s3fs.S3FileSystem()
    with fs.open(path, mode="rb") as f:
        content = f.read()
    return content


def get_content_from_url(url: str) -> BytesIO:
    """
    Get BytesIO stream from URL.
    Args
        url (str): URL
    Returns
        io.BytesIO: content of URL as BytesIO stream
    """
    logging.info(f"Loading file from URL: {url}")
    with requests.Session() as session:
        res = session.get(url)
    content = BytesIO(res.content)
    return content


def get_df_from_parquet_s3_path(path: str, **kwargs) -> pl.DataFrame:
    """
    Get dataframe from Parquet file stored in s3 path.

    Args
        path (str): S3 URI to Parquet file
        **kwargs for pl.read_parquet()
    Returns
        pl.DataFrame: dataframe from Parquet file
    """
    content = BytesIO(get_content_from_s3_path(path))
    df = pl.read_parquet(content, **kwargs)
    return df


def get_gdf_from_gpkg_s3_path(path: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Get GeoDataFrame from GeoPackage file stored in s3 path.

    Args
        path (str): S3 URI to GeoPackage file
        **kwargs for gpd.read_file()
    Returns
        gpd.GeoDataFrame: geodataframe from GeoPackage file
    """
    content = BytesIO(get_content_from_s3_path(path))
    gdf = gpd.read_file(content, **kwargs)
    return gdf


def load_pickle(path: str) -> Any:
    """
    Load the content of a pickle file.

    Args:
        path (str): path to pickle file

    Returns:
        Any: contents of pickle file
    """
    fs = s3fs.S3FileSystem()
    with fs.open(path, "rb") as file:
        return pickle.load(file)
