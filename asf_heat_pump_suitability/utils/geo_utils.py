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
    binary_data: Union[BaseGeometry, bytes, str]
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
    la_lsoas: List[str], lsoa_shp_path: str, target_crs: str
) -> gpd.GeoDataFrame:
    """
    Loads and preprocesses LSOA geospatial data by:
      - Reading a shapefile.
      - Converting to a target CRS if necessary.
      - Filtering LSOAs to just those that belong to the specified local authority.

    Args:
        la_lsoas (List[str]): List of LSOA codes for the current local authority.
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

    la_lsoa_geometries_gdf = lsoa_gdf[lsoa_gdf["LSOA21CD"].isin(la_lsoas)]
    if la_lsoa_geometries_gdf.empty:
        raise ValueError(
            "No LSOAs found for this local authority in the provided LSOA shapefile."
        )

    return la_lsoa_geometries_gdf


def merge_hp_suitability_data_with_geometries(
    hp_suitability_scores_pd: pd.DataFrame,
    la_lsoa_geometries_gdf: gpd.GeoDataFrame,
    la_name: str,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """
    Merges the LA's heat pump suitability scores (pandas DataFrame) with its LSOA geometries (GeoDataFrame).
    Ensures the final merged result is also a GeoDataFrame in the target CRS.

    Args:
        hp_suitability_scores_pd (pd.DataFrame): DataFrame containing columns like 'LSOA21CD' or 'DESNZ_pilot_fraction'.
        la_lsoa_geometries_gdf (gpd.GeoDataFrame): LSOA geometry data for the LA.
        la_name (str): Local Authority name, used for logging.
        target_crs (str): The target Coordinate Reference System (e.g., "EPSG:27700").

    Returns:
        gpd.GeoDataFrame: Merged data with geometry.

    Raises:
        ValueError: If the merge results in empty data or missing geometry.
    """
    logging.info(f"Merging suitability data with geometries for {la_name}...")
    merged = hp_suitability_scores_pd.merge(
        la_lsoa_geometries_gdf, on="LSOA21CD", how="left"
    )

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
