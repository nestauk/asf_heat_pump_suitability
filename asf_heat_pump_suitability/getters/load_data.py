import polars as pl
import pandas as pd
import geopandas as gpd
import logging
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, schemas
from io import StringIO
from tenacity import retry, stop_after_attempt
import warnings
import pyogrio
from typing import Tuple

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
    logging.info("Loading Microsoft building footprint data-links file")
    df = pd.read_csv(
        config["data_source"]["global_microsoft_building_footprint_links"],
        dtype=schemas.microsoft_datalinks,
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


# TODO move to utils for flats on fossils
def get_df_osopen_uprn_latlon(**kwargs) -> pl.DataFrame:
    """
    Get raw OS (Ordnance Survey) Open UPRN dataset containing latitude and longitude and British National Grid x and y
    coordinates for all UPRNs in Great Britain.

    Args:
        **kwargs fo pl.read_csv

    Returns:
        pl.DataFrame: raw OS Open UPRN dataset with lat/lon and x/y coordinates for every UPRN
    """
    print("Loading OS OpenMap UPRN dataset...")
    df = base_getters.get_df_from_zip_csv_s3(
        config["data_source"]["GB_osopen_uprn_latlon"],
        extract_file="osopenuprn_202405.csv",
        **kwargs,
    )

    return df


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


def load_df_dz_lookup(**kwargs) -> pl.DataFrame:
    """
    Load Data Zone to LAD lookup table from Scottish Government.

    Args:
        **kwargs for `polars.read_csv()`

    Returns:
        pl.DataFrame: DZ to LAD lookup table for Scotland
    """
    df = pl.read_csv(
        config["data_source"]["S_dz_lookup"],
        infer_schema_length=5000,
        ignore_errors=True,
        **kwargs,
    )
    return df
