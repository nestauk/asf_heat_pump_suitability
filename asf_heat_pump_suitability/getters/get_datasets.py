import polars as pl
import pandas as pd
import geopandas as gpd
import logging
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters, schemas


def get_df_ons_pd(**kwargs) -> pl.DataFrame:
    """
    Get ONS postcode directory (ONSPD) for Great Britain.

    Args:
        **kwargs for pl.read_csv

    Returns:
        pl.DataFrame: postcode directory for Great Britain
    """
    df = base_getters.get_df_from_zip_url(
        url=config["data_source"]["gb_ons_postcode_dir_url"],
        extract_file=config["data_source"]["gb_ons_postcode_dir_file_path"],
        schema=schemas.onspd_schema,
        **kwargs,
    )

    return df


def load_gdf_ons_council_bounds() -> gpd.GeoDataFrame:
    """
    Load ONS council bounding polygons for the UK (CRS: EPSG:4326).

    Returns:
        gpd.GeoDataFrame: ONS councils with bounding polygons
    """
    gdf = base_getters.load_gdf_from_s3_geojson(
        config["data_source"]["UK_ons_lad_bounds"], crs="WGS84"
    )

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
    gdf = base_getters.load_gdf_csv_gz(url)

    return gdf


def load_gdf_inspire_land_parcels(path: str) -> gpd.GeoDataFrame:
    """
    Load land registry's index polygons spatial data (INSPIRE) showing the geometry and extent of registered freehold
    properties in England and Wales. CRS EPSG:27700, British National Grid.

    Args:
        path (str): path to INSPIRE land parcel file

    Returns:
        gpd.GeoDataFrame: registered land extent polygons for one council
    """
    logging.info(f"Loading INSPIRE land parcel file: {path}")
    gdf = gpd.read_file(path, engine="pyogrio")

    return gdf


def get_df_ons_garden_space_avg(**kwargs) -> pl.DataFrame:
    """
    Get raw ONS 'Access to garden space, Great Britain' dataset.

    Args:
        **kwargs for pl.read_excel

    Returns:
        pl.DataFrame: raw ONS 'Access to garden space' dataset
    """
    content = base_getters.get_content_from_s3_path(
        config["data_source"]["GB_ons_garden_space_access"]
    )
    df = pl.read_excel(
        content,
        sheet_name="MSOA gardens",
        engine="calamine",
        **kwargs,
    )
    return df


def get_df_osopen_uprn_latlon(**kwargs) -> pl.DataFrame:
    """
    Get raw OS (Ordnance Survey) Open UPRN dataset containing latitude and longitude and British National Grid x and y
    coordinates for all UPRNs in Great Britain.

    Args:
        **kwargs fo pl.read_csv

    Returns:
        pl.DataFrame: raw OS Open UPRN dataset with lat/lon and x/y coordinates for every UPRN
    """
    df = base_getters.get_df_from_zip_csv_s3(
        config["data_source"]["GB_osopen_uprn_latlon"],
        extract_file="osopenuprn_202405.csv",
        **kwargs,
    )

    return df


def load_gdf_historic_england_conservation_areas() -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with conservation area polygons from Historic England (CRS: EPSG:4326).

    Returns:
        gpd.GeoDataFrame: polygons of conservation areas in England
    """
    gdf = gpd.read_file(
        config["data_source"]["E_historic_england_conservation_areas"],
        engine="pyogrio",
    )

    return gdf
