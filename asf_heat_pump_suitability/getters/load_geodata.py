import polars as pl
import geopandas as gpd
import regex as re
import os

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


def load_df_osopen_uprn(**kwargs) -> pl.DataFrame:
    """
    Get raw OS (Ordnance Survey) Open UPRN dataset containing latitude and longitude and British National Grid X and Y
    coordinates for all UPRNs in Great Britain.

    Args:
        **kwargs for pl.read_csv

    Returns:
        pl.DataFrame: raw OS Open UPRN dataset with lat/lon and x/y coordinates for every UPRN
    """
    print("Loading OSOpen UPRNs...")
    path = config["data"]["geodata"]["uk_osopen_uprn"]
    filename = os.path.basename(path).split("_csv")[0]
    df = base_getters.get_df_from_zip_csv_s3(
        path,
        extract_file=f"{filename}.csv",
        **kwargs,
    )

    return df


def load_gdf_plymouth_heat_network_zone_areas(**kwargs) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame with Plymouth heat network zone polygons from Plymouth City Council (CRS: EPSG:27700).

    Args:
        **kwargs for `gpd.read_file()`

    Returns:
        gpd.GeoDataFrame: polygons of heat network zones in Plymouth.
    """
    gdf = base_getters.get_gdf_from_gpkg_s3_path(
        path=config["data"]["geodata"]["heat_network_zones"]["plymouth"],
        **kwargs,
    )
    return gdf
