"""
Functions to filter OS UPRNs to UPRNs located within heat network zones.
"""

import geopandas as gpd

from asf_heat_pump_suitability import config


def filter_gdf_heat_network_zone_uprns(
    uprn_gdf: gpd.GeoDataFrame,
    hn_zone_gdf: gpd.GeoDataFrame,
    hn_zone_usecols: list = None,
) -> gpd.GeoDataFrame:
    """
    Filter UPRNs to UPRNs that are located within a heat network zone.

    Args:
        uprn_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be filtered
        hn_zone_gdf (gpd.GeoDataFrame): polygons of heat network zones

    Returns:
        gpd.GeoDataFrame: UPRNs located within any of the heat network zones
    """

    # CRS checks
    assert (
        uprn_gdf.crs == config["constant"]["target_crs"]
    ), f"Target CRS is ESPG:{config['constant']['target_crs']}, uprn_gdf is {uprn_gdf.crs}"

    assert (
        uprn_gdf.crs == hn_zone_gdf.crs
    ), f"CRS mismatch: uprn_gdf is {uprn_gdf.crs} and hn_zone_gdf is {hn_zone_gdf.crs}"

    # If hn_zone_usecols not specified, use all columns in hn_zone_gdf
    hn_zone_gdf = hn_zone_gdf if not hn_zone_usecols else hn_zone_gdf[hn_zone_usecols]

    filtered_uprn_gdf = uprn_gdf.sjoin(
        hn_zone_gdf, how="inner", predicate="intersects"
    ).drop(columns="index_right")

    return filtered_uprn_gdf
