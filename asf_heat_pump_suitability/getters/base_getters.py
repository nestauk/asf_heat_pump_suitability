import requests
import polars as pl
from zipfile import ZipFile
from io import BytesIO
import logging
import s3fs
import geojson
import geopandas as gpd


def get_df_from_excel_url(url: str, **kwargs) -> pl.DataFrame:
    """
    Get dataframe from Excel file stored at URL.

    Args
        url (str): URL location of Excel file download
        **kwargs for pl.read_excel()

    Returns
        pl.DataFrame: dataframe from Excel file
    """
    content = _get_content_from_url(url)
    df = pl.read_excel(content, **kwargs)

    return df


def get_df_from_zip_url(url: str, extract_file: str, **kwargs) -> pl.DataFrame:
    """
    Get dataframe from ZIP file stored at URL.

    Args
        url (str): URL location of ZIP file download
        extract_file (str): name of file to extract
        **kwargs for pl.read_csv()

    Returns
        pl.DataFrame: dataset from ZIP file
    """
    content = _get_content_from_url(url)
    df = pl.read_csv(ZipFile(content).open(name=extract_file), **kwargs)

    return df


def get_content_from_path(path: str) -> bytes:
    """
    Get bytes content of file from path.

    Args
        path (str): path to file

    Returns
        bytes: bytes content of file
    """
    fs = s3fs.S3FileSystem()
    with fs.open(path, mode="rb") as f:
        content = f.read()
    return content


def _get_content_from_url(url: str) -> BytesIO:
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


def load_gdf_from_s3_geojson(s3_uri: str) -> gpd.GeoDataFrame:
    """ """
    fs = s3fs.S3FileSystem()
    with fs.open(s3_uri, "rb") as f:
        data = geojson.load(f)
    gdf = gpd.GeoDataFrame.from_features(data["features"])

    return gdf


def list_files_s3_location(location: str) -> list:
    """ """
    fs = s3fs.S3FileSystem()
    files = fs.ls(location)

    return files
