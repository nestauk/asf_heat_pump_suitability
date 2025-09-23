import logging
from typing import Union, List

import geopandas as gpd
import pandas as pd
import shapely

from shapely.geometry.base import BaseGeometry
from shapely import wkb


def transform_gdf_drop_duplicates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Drop polygons with the same representative point. This drops both duplicate and nearly identical geometries.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with polygon geometries

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with duplicate polygon geometries dropped
    """
    gdf["rep_point"] = gdf.representative_point().to_wkb()
    if gdf["rep_point"].nunique() != len(gdf):
        duplicate_count = gdf.duplicated(subset="rep_point").sum()

        logging.info(
            f"Polygons containing same representative point found. "
            f"Dropping {duplicate_count} polygons."
        )

        # Sort values to create replicable duplicate removal process
        gdf = gdf.sort_values(by="geometry").drop_duplicates(
            subset="rep_point", keep="first"
        )

    gdf = gdf.drop("rep_point", axis=1)

    return gdf


def get_polygon_gdf_bounds(gdf: gpd.GeoDataFrame) -> shapely.Polygon:
    """
    Get bounding polygon of GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame

    Returns:
        shapely.Polygon: bounding polygon of GeoDataFrame
    """
    return shapely.box(*gdf.total_bounds)


def parse_binary_geometry(
    binary_data: Union[BaseGeometry, bytes, str],
) -> Union[BaseGeometry, None]:
    """
    Parse binary geometry data into Shapely geometry object.

    Args:
        binary_data: Input geometry data in various formats.

    Returns:
        Shapely geometry object or None if parsing fails.
    """
    if isinstance(binary_data, BaseGeometry):
        return binary_data
    elif isinstance(binary_data, bytes):
        return wkb.loads(binary_data)
    elif isinstance(binary_data, str):
        return wkb.loads(binary_data, hex=True)
    else:
        return None
