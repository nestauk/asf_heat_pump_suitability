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
    content = get_content_from_url(url)
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
    content = get_content_from_url(url)
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


def load_gdf_from_s3_geojson(s3_uri: str, crs: str) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame from GeoJSON on S3.

    Args:
        s3_uri (str): URI to S3 GeoJSON
        crs (str): coordinate reference system of GeoJSON

    Returns:
        gpd.GeoDataFrame
    """
    fs = s3fs.S3FileSystem()
    with fs.open(s3_uri, "rb") as f:
        data = geojson.load(f)
    gdf = gpd.GeoDataFrame.from_features(data["features"], crs=crs)

    return gdf


def list_obj_s3_location(location: str) -> list:
    """
    List objects in an S3 location.

    Args:
        location (str): S3 URI

    Returns:
        list: objects in S3 location
    """
    fs = s3fs.S3FileSystem()
    o = fs.ls(location)

    return o


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


def get_df_from_parquet(path: str, is_s3: bool = False, **kwargs) -> pl.DataFrame:
    """
    Get a Polars dataframe from a Parquet file, either from local disk or from an S3 URI.

    Args:
        path (str):
            - If is_s3=False, this is a *local file path* to the Parquet file, e.g. "./data/foo.parquet"
            - If is_s3=True, this is an *S3 URI*, e.g. "s3://my-bucket/path/foo.parquet"
        is_s3 (bool): Whether to interpret 'path' as an S3 URI or a local path.
        **kwargs: Additional keyword args for pl.read_parquet().

    Returns:
        pl.DataFrame: Loaded Parquet file as a Polars DataFrame.
    """
    if is_s3:
        # We use your existing utility that fetches the raw bytes from S3
        polars_df = get_content_from_s3_path(path, **kwargs)  # returns bytes
        # Convert to an in-memory buffer and read with Polars
        return polars_df
    else:
        # Read directly from local file system
        print("ISSUE IDENTIFIED")
        return pl.read_parquet(path, **kwargs)
