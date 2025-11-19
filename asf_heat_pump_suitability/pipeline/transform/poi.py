"""
Functions to preprocess Points of Interest (POI) data for Great Britain.
"""

import geopandas as gpd
from typing import Iterable
import csv
import smart_open

from asf_heat_pump_suitability import config


def transform_gdf_poi(
    poi: gpd.GeoDataFrame,
    filter_categories: Iterable[str] = None,
    target_crs: str | int = config["constant"]["target_crs"],
) -> gpd.GeoDataFrame:
    """
    Transform Points of Interest data for Great Britain - filter to specified categories and reproject to target CRS.

    Args:
        poi (gpd.GeoDataFrame): Points Of Interest dataset
        filter_categories (List[str]): types of POI to filter the dataframe for from the `main_category` column. Default None to load all categories.
        target_crs (str | int): coordinate reference system to reproject data to

    Returns:
        gpd.GeoDataFrame: Processed POI data containing types of POI specified
    """
    # Filter to Great Britain
    poi = poi[poi.country == "GB"].copy()

    # Filter anchor properties and reproject
    if filter_categories:
        poi = poi[poi.main_category.isin(filter_categories)]
    poi = poi.to_crs(target_crs)

    print(f"Found {len(poi)} points of interest")
    print(f'POI CRS converted to: {config["constant"]["target_crs"]}')
    return poi


def load_set_non_domestic_poi_categories() -> set:
    """
    Load set of Points of Interest categories which are unlikely to be located in buildings with domestic properties.

    Returns:
        set: non-domestic POI categories
    """
    with smart_open.open(
        config["data"]["processed"]["non_domestic_poi_categories"], "r"
    ) as f:
        categories = [line.strip() for line in f]
    return set(categories)
