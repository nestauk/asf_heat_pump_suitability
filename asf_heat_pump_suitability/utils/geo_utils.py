import logging
from typing import Union, List

import geopandas as gpd
import pandas as pd
import shapely

from shapely.geometry.base import BaseGeometry
from shapely import wkb
from asf_heat_pump_suitability import config


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


def verify_gdf_crs(
    gdf: gpd.GeoDataFrame, target_crs: str | int = config["constant"]["target_crs"]
) -> gpd.GeoDataFrame:
    """
    Verify GeoDataFrame uses the defined target CRS. If not, convert to target CRS.

    Args:
        gdf (gpd.GeoDataFrame): geodataset to check CRS of
        target_crs (str|int): target CRS. Defaults to target CRS defined in config base.yaml

    Returns:
        gpd.GeoDataFrame: geodataset in target CRS
    """
    if gdf.crs != target_crs:
        print(f"Converting GeoDataFrame to target CRS: {target_crs}")
        return gdf.to_crs(target_crs)
    else:
        return gdf


def find_gdf_overlapping_geometries(
    gdf: gpd.GeoDataFrame, return_geometries: bool = True, id_col=None
) -> gpd.GeoDataFrame | set:
    """
    Find overlapping geometries within the same GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): containing geometries to check for overlaps
        return_geometries (bool): set to `True` to return a `GeoDataFrame` with the overlapping geometries. Set to `False`
        to return just the IDs of overlapping geometries.
        id_col (str): name of unique ID for geometries. Required only when `return_geometries` is set to `False`. Default
        `None` to use index.

    Returns:
        gpd.GeoDataFrame | set: geometries or IDs of geometries which overlap with any other geometry within the `GeoDataFrame`
    """
    buffer = gdf.copy()
    # Buffer added to prevent touching neighbours being identified as overlapping
    buffer["geometry"] = buffer["geometry"].buffer(-0.0001)
    intersecting_gdf = buffer.sjoin(buffer, predicate="intersects")
    intersecting_gdf = intersecting_gdf.loc[
        intersecting_gdf.index != intersecting_gdf.index_right
    ]
    if return_geometries:
        return intersecting_gdf
    else:
        if id_col:
            overlapping_ids = set(intersecting_gdf[f"{id_col}_left"])
        else:
            overlapping_ids = set(intersecting_gdf.index)
            id_col = "index"

        print(f"ID used: {id_col}. Overlapping ID count: {len(overlapping_ids)}")
        return overlapping_ids
