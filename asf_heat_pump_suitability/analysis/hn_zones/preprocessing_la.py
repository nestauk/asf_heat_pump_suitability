"""
This module provides preprocessing utilities for Local Authority (LA) heat network zone (HNZ) analysis.
It includes functions to:

1. Ensure matching Coordinate Reference Systems (CRS) between GeoDataFrames.
2. Load and filter LSOA geometries based on specified codes.
3. Merge heat network suitability scores with LSOA geometries.

These functions are essential for preparing geospatial data for further analysis and visualization.
"""

import logging
from typing import List
import geopandas as gpd
import pandas as pd


def ensure_crs_match(
    gdf_1: gpd.GeoDataFrame, gdf_2: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """ "
    Ensure that the CRS of both GeoDataFrames match.

    Args:
        gdf_1 (gpd.GeoDataFrame)
        gdf_2 (gpd.GeoDataFrame)

    Returns:
        gpd.GeoDataFrame: GeoDataFrame 2 reprojected to match GeoDataFrame 1 if necessary.
    """
    if gdf_1.crs != gdf_2.crs:
        gdf_2 = gdf_2.to_crs(gdf_1.crs)
    return gdf_2


def load_and_filter_lsoa_geometries(
    lsoas: List[str], lsoa_shp_path: str, target_crs: str
) -> gpd.GeoDataFrame:
    """
    Loads and preprocesses LSOA geospatial data by:
      - Reading a shapefile.
      - Converting to a target CRS if necessary.
      - Filtering LSOAs to just those specified.

    Args:
        lsoas (List[str]): List of LSOA codes to get geometries for.
        lsoa_shp_path (str): Path (local or S3) to the national LSOA shapefile.
        target_crs (str): The target Coordinate Reference System (e.g., "EPSG:27700").

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with filtered LSOA geometries in the target CRS.

    Raises:
        ValueError: If no matching LSOAs are found in the shapefile.
    """
    logging.info("Preprocessing LSOA geometries...")
    lsoa_gdf = gpd.read_file(lsoa_shp_path)

    # Reproject if needed
    if lsoa_gdf.crs != target_crs:
        logging.info(f"Converting shapefile to {target_crs}")
        lsoa_gdf = lsoa_gdf.to_crs(target_crs)

    lsoa_gdf = lsoa_gdf[lsoa_gdf["LSOA21CD"].isin(lsoas)]
    if lsoa_gdf.empty:
        raise ValueError("Specified LSOAs not found in the provided LSOA shapefile.")

    return lsoa_gdf


def merge_hp_suitability_data_with_geometries(
    hn_scores_pd: pd.DataFrame,
    la_lsoa_geometries_gdf: gpd.GeoDataFrame,
    la_name: str,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """
    Merges the LA's heat network suitability scores (pandas DataFrame) with its LSOA geometries (GeoDataFrame).
    Ensures the final merged result is also a GeoDataFrame in the target CRS.

    Args:
        hn_scores_pd (pd.DataFrame): DataFrame containing columns like 'LSOA21CD' and "DESNZ_pilot_fraction".
        la_lsoa_geometries_gdf (gpd.GeoDataFrame): LSOA geometry data for the LA.
        la_name (str): Local Authority name, used for logging.
        target_crs (str): The target Coordinate Reference System (e.g., "EPSG:27700").

    Returns:
        gpd.GeoDataFrame: Merged data with geometry.

    Raises:
        ValueError: If the merge results in empty data or missing geometry.
    """
    logging.info(f"Merging suitability data with geometries for {la_name}...")
    merged = hn_scores_pd.merge(la_lsoa_geometries_gdf, on="LSOA21CD", how="left")

    if merged.empty:
        raise ValueError(
            f"Merging resulted in an empty DataFrame for {la_name}. Check LSOA codes."
        )
    if merged["geometry"].isna().any():
        raise ValueError(
            f"Some records are missing geometry after merging for {la_name}."
        )

    la_hp_suitability_gdf = gpd.GeoDataFrame(
        merged, geometry="geometry", crs=target_crs
    )
    return la_hp_suitability_gdf
