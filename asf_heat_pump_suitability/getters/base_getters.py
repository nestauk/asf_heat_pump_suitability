import requests
import polars as pl
from zipfile import ZipFile
from io import BytesIO
import logging
from asf_heat_pump_suitability import config


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
