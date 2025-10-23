import geopandas as gpd
from typing import List

from asf_heat_pump_suitability import config


def transform_gdf_poi(
    poi: gpd.GeoDataFrame,
    filter_categories: List[str] = None,
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
    poi = poi.drop_duplicates(subset="geometry", keep="first").to_crs(target_crs)

    print(f"Found {len(poi)} points of interest")
    print(f'POI CRS converted to: {config["constant"]["target_crs"]}')
    return poi
