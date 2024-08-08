import requests
import polars as pl
from zipfile import ZipFile
from io import BytesIO
import logging
import s3fs


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

    Args:
        url (str): URL location of ZIP file load
        extract_file (str): name of file to extract
        **kwargs for pl.read_csv()

    Returns:
        pl.DataFrame: dataset from ZIP file
    """
    content = _get_content_from_url(url)
    df = pl.read_csv(ZipFile(content).open(name=extract_file), **kwargs)

    return df


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
