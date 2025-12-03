"""
Functions to filter UPRNs to those located within heat network zones.
"""

import geopandas as gpd
import polars as pl

from asf_heat_pump_suitability import config


def filter_gdf_heat_network_zone_uprns(
    uprn_gdf: gpd.GeoDataFrame,
    hn_zone_gdf: gpd.GeoDataFrame,
    usecols: list = None,
) -> pl.DataFrame:
    """
    Filter UPRNs to UPRNs that are located within a heat network zone.

    Args:
        uprn_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be filtered
        hn_zone_gdf (gpd.GeoDataFrame): polygons of heat network zones
        usecols: names of descriptive columns in hn_zone_gdf (excluding "geometry") to be joined with uprn_gdf. Default is None to use all columns.

    Returns:
        pl.DataFrame: input UPRNs labelled with heat network zone identifiers
    """

    # CRS checks and reprojection if needed
    target_crs = config["constant"]["target_crs"]

    if uprn_gdf.crs != target_crs:
        uprn_gdf = uprn_gdf.to_crs(target_crs)
        print(f"uprn_gdf reprojected to target CRS: {target_crs}")

    if hn_zone_gdf.crs != target_crs:
        hn_zone_gdf = hn_zone_gdf.to_crs(target_crs)
        print(f"hn_zone_gdf reprojected to target CRS: {target_crs}")

    # If usecols specified, filter columns in hn_zone_gdf
    if usecols:
        hn_zone_gdf = hn_zone_gdf[["geometry"] + usecols]

    # Spatial join for filtering
    filtered_uprn_gdf = uprn_gdf.sjoin(
        hn_zone_gdf,
        how="inner",
        predicate="intersects",  # include properties intersecting heat network zone boundary
    ).drop(columns="index_right")

    # Add heat network zone boolean label to original uprn_gdf
    uprn_gdf["in_hn_zone"] = uprn_gdf["UPRN"].isin(filtered_uprn_gdf["UPRN"])

    # Add description labels from hn_zone_gdf to original uprn_gdf
    label_cols = [col for col in hn_zone_gdf.columns if col != "geometry"]
    uprn_gdf = uprn_gdf.merge(
        filtered_uprn_gdf[["UPRN"] + label_cols], on="UPRN", how="left"
    )

    # Return as polars df without geometry
    uprn_df = pl.from_pandas(uprn_gdf.drop(columns="geometry"))

    return uprn_df
