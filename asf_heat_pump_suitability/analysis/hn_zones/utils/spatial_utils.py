"""
Spatial Utility Functions for Heat Network and LSOA Data Processing

This module provides utility functions for:
- Loading DESNZ heat network zones and LSOA polygons.
- Ensuring CRS consistency between datasets.
- Performing spatial intersections to determine heat network coverage.
- Computing the fraction of LSOA areas covered by heat network zones.

**Functions:**
- `_load_geodata`: Loads heat network and LSOA data.
- `_ensure_crs_match`: Ensures CRS consistency between GeoDataFrames.
- `_compute_lsoa_coverage_stats`: Computes spatial intersections and LSOA coverage stats.
- `load_transform_hn_geodata`: High-level function to execute all steps in sequence.

This module is used in the main processing script to analyse spatial relationships
between heat network zones and local authority areas.
"""

from typing import Tuple, List
import geopandas as gpd
import pyogrio


def _load_geodata(
    gpkg_path: str, shp_path: str, layer_name: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load DESNZ heat network polygons from a GeoPackage and LSOA shapefile.

    Args:
        desnz_hn_gpkg_path (str): Path to the DESNZ Heat Network GeoPackage.
        lsoa_shp_path (str): Path to the LSOA shapefile.
        layer_name (str): Layer name in the GeoPackage.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            - DESNZ heat network zones as a GeoDataFrame.
            - LSOA polygons as a GeoDataFrame.
    """
    hn_gdf = pyogrio.read_dataframe(gpkg_path, layer=layer_name)
    lsoa_gdf = gpd.read_file(shp_path)
    return hn_gdf, lsoa_gdf


def _ensure_crs_match(
    desnz_hn_gdf: gpd.GeoDataFrame, lsoa_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """ "
    Ensure that the CRS of both GeoDataFrames match.

    Args:
        desnz_hn_gdf (gpd.GeoDataFrame): DESNZ heat network zones.
        lsoa_gdf (gpd.GeoDataFrame): LSOA polygons.

    Returns:
        gpd.GeoDataFrame: LSOA GeoDataFrame reprojected to match DESNZ if necessary.
    """
    if desnz_hn_gdf.crs != lsoa_gdf.crs:
        lsoa_gdf = lsoa_gdf.to_crs(desnz_hn_gdf.crs)
    return lsoa_gdf


def _compute_lsoa_coverage_stats(
    desnz_hn_gdf: gpd.GeoDataFrame, lsoa_gdf: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """ "
    Perform a spatial intersection between DESNZ heat network zones and LSOA polygons,
    then calculate the fraction of each LSOA's area that is covered by heat network zones.

    Args:
        desnz_hn_gdf (gpd.GeoDataFrame): DESNZ heat network zones.
        lsoa_gdf (gpd.GeoDataFrame): LSOA polygons.

    Returns:
        Tuple[gpd.GeoDataFrame, List[str]]:
            - GeoDataFrame containing LSOA codes, intersection geometry, and fraction_covered.
            - List of unique LSOA codes present in the final GeoDataFrame.
    """
    # Remove 'index_right' column if it exists
    if "index_right" in desnz_hn_gdf.columns:
        desnz_hn_gdf = desnz_hn_gdf.drop(columns=["index_right"])

    # Ensure LSOAs have a valid total area column
    lsoa_gdf["total_area"] = lsoa_gdf.geometry.area

    # Perform intersection to determine which LSOAs overlap with heat network zones
    intersection_gdf = gpd.overlay(desnz_hn_gdf, lsoa_gdf, how="intersection")

    # Calculate the fraction of the LSOA that is covered by heat network zones
    intersection_gdf["fraction_covered"] = (
        intersection_gdf["geometry"].area / intersection_gdf["total_area"]
    )

    # Extract unique LSOA codes
    unique_lsoa_codes = intersection_gdf["LSOA21CD"].dropna().unique().tolist()

    return intersection_gdf, unique_lsoa_codes


def load_transform_hn_geodata(
    desnz_hn_gpkg_path: str, lsoa_shp_path: str, layer_name: str
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Load the Heat Networks GeoPackage file, ensure consistent CRS,
    perform a spatial join with LSOA polygons, and calculate the intersection area.

    Args:
        desnz_hn_gpkg_path (str): Path to the GeoPackage file with the Heat Network zones.
        lsoa_shp_path (str): Path to the LSOA shapefile.
        layer_name (str): Layer name in the GeoPackage.

    Returns:
        Tuple[gpd.GeoDataFrame, List[str]]:
            - GeoDataFrame with LSOA codes added via spatial join.
            - List of unique LSOA codes present in the GeoDataFrame.
    """
    desnz_hn_gdf, lsoa_gdf = _load_geodata(
        desnz_hn_gpkg_path, lsoa_shp_path, layer_name
    )
    lsoa_gdf = _ensure_crs_match(desnz_hn_gdf, lsoa_gdf)
    return _compute_lsoa_coverage_stats(desnz_hn_gdf, lsoa_gdf)
