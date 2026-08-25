"Functions to load non-geospatial datasets. No or minimal processing occurs in these functions."

from collections import OrderedDict

import polars as pl
import pandas as pd
import geopandas as gpd
import logging

import boto3

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.utils import s3_utils
import warnings

from typing import Optional, List

# Ignore RunTimeWarning when loading Microsoft building footprint files
# as reading from gzipped stream should be faster than unzipping and loading data
warnings.filterwarnings("ignore", category=RuntimeWarning, message="VSIFSeekL")


def load_gdf_ons_council_bounds(**kwargs) -> gpd.GeoDataFrame:
    """
    Load ONS council bounding polygons for the UK (CRS: EPSG:27700).

    Args:
        **kwargs for gpd.read_file

    Returns:
        gpd.GeoDataFrame: ONS councils with bounding polygons
    """
    gdf = gpd.read_file(config["data_source"]["UK_ons_lad_bounds"], **kwargs)

    return gdf


def load_df_microsoft_building_footprint_links() -> pd.DataFrame:
    """
    Load Microsoft Global ML Building Footprints data links file containing URLs to all building footprint files
    available.

    Returns:
        pd.DataFrame: Microsoft Global ML Building Footprints data links
    """
    schema = OrderedDict(
        [("Location", str), ("QuadKey", str), ("Url", str), ("Size", str)]
    )
    logging.info("Loading Microsoft building footprint data-links file")
    df = pd.read_csv(
        config["data_source"]["global_microsoft_building_footprint_links"],
        dtype=schema,
    )

    return df


def load_gdf_microsoft_building_footprints(url: str) -> gpd.GeoDataFrame:
    """
    Load Microsoft building footprints file (CRS: EPSG:4326).

    Args:
        url (str): URL to Microsoft building footprint file

    Returns:
        gpd.GeoDataFrame: Microsoft building footprint polygons
    """
    gdf = gpd.read_file(
        f"GeoJSONSeq:/vsigzip//vsicurl/{url}", engine="pyogrio", use_arrow=True
    )

    return gdf


def load_df_off_gas_pcds() -> pl.DataFrame:
    """
    Get off gas grid postcodes from Supply Point Administration dataset.

    Returns:
        pl.DataFrame: raw off gas grid dataset
    """
    df = base_getters.get_df_from_excel_s3_path(
        config["data_source"]["UK_spa_offgasgrid"], sheet_name="Off-Gas Postcodes 2024"
    )
    return df


def load_df_uprn_lookup(
    grid_squares: Optional[List[str]] = None, **kwargs
) -> pl.DataFrame:
    """
    Load UPRN national statistics lookup for all of GB or a given list of grid squares if specified.

    Args:
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded. Default None to load whole GB.
        **kwargs for polars.read_parquet()

    Returns:
        pl.DataFrame: UPRN national statistics lookup for specified area(s)
    """
    print("Loading UPRN national statistics lookup...")
    uri = config["data"]["geodata"]["gb_uprn_lookup_partitioned"]
    if grid_squares:
        return pl.concat(
            [
                pl.read_parquet(uri.format(grid_square=grid_square), **kwargs)
                for grid_square in grid_squares
            ]
        )
    else:  # Load whole of GB
        bucket, prefix = s3_utils.extract_tuple_bucket_prefix(uri)
        fs = boto3.client("s3")
        files = s3_utils.fetch_list_file_paths_from_s3_folder(
            s3_client=fs,
            s3_bucket=bucket,
            path_folder=prefix,
            file_type=".parquet",
        )

        return pl.concat([pl.read_parquet(file, **kwargs) for file in files])


def load_df_domestic_epc(grid_squares: Optional[List[str]], **kwargs) -> pl.DataFrame:
    """
    Load processed domestic EPC data (processed with asf-daps) for given grid squares or all of Great Britain.

    Args:
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded. Default None to load whole GB.
        **kwargs for polars.read_parquet()

    Returns:
        pl.DataFrame: domestic EPC data
    """
    print("Loading domestic EPC data...")
    if grid_squares:
        uri = config["data"]["epc"]["domestic_partitioned"]
        return pl.concat(
            [
                pl.read_parquet(uri.format(grid_square=grid_square), **kwargs)
                for grid_square in grid_squares
            ]
        )
    else:
        return pl.read_parquet(config["data"]["epc"]["domestic"], **kwargs)
