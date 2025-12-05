"""
Functions to label UPRNs within existing or potential heat network zones.
"""

import geopandas as gpd
import polars as pl

from asf_heat_pump_suitability import config


def label_gdf_heat_network_zone_uprns(
    uprn_gdf: gpd.GeoDataFrame,
    hn_zone_gdf: gpd.GeoDataFrame,
    usecols: list = None,
) -> pl.DataFrame:
    """
    Label UPRNs that are located within a heat network zone.

    Args:
        uprn_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be labelled
        hn_zone_gdf (gpd.GeoDataFrame): polygons of heat network zones
        usecols (list, optional): names of descriptive columns in hn_zone_gdf to be joined with uprn_gdf (in addition to "geometry").
            Default is None to only use "geometry".

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
    else:
        hn_zone_gdf = hn_zone_gdf[["geometry"]]

    # Spatial join for labelling UPRNs
    labelled_uprn_gdf = uprn_gdf.sjoin(
        hn_zone_gdf,
        how="left",
        predicate="intersects",  # include properties intersecting heat network zone boundary
    ).drop(columns="index_right")

    # Add heat network zone boolean label
    labelled_uprn_gdf["in_hn_zone"] = labelled_uprn_gdf["ZoneID"].notna()

    # Return as polars df without geometry
    labelled_uprn_df = pl.from_pandas(labelled_uprn_gdf.drop(columns="geometry"))

    return labelled_uprn_df


def label_gdf_heat_network_spatial_signatures_uprns(
    uprn_gdf: gpd.GeoDataFrame,
    spatial_signatures_gdf: gpd.GeoDataFrame,
    usecols: list = None,
    types: list = None,
) -> pl.DataFrame:
    """
    Labels UPRNs that are located within a given Spatial Signature type deemed to be potentially
    suitable for a heat network.

    Args:
        uprn_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be labelled
        spatial_signatures_gdf (gpd.GeoDataFrame): polygons of spatial signatures
        usecols (list, optional): names of descriptive columns from spatial_signatures_gdf to be joined with uprn_gdf (in addition to "geometry" and "type").
            Default is None to use only "geometry" and "type".
        types (list, optional): spatial signature types potentially suitable for a heat network. Default is None to use all spatial signature types.

    Returns:
        pl.DataFrame: input UPRNs labelled with spatial signature identifiers and booleans indicating potential for a heat network
    """

    # CRS checks and reprojection if needed
    target_crs = config["constant"]["target_crs"]

    if uprn_gdf.crs != target_crs:
        uprn_gdf = uprn_gdf.to_crs(target_crs)
        print(f"uprn_gdf reprojected to target CRS: {target_crs}")

    if spatial_signatures_gdf.crs != target_crs:
        spatial_signatures_gdf = spatial_signatures_gdf.to_crs(target_crs)
        print(f"spatial_signatures_gdf reprojected to target CRS: {target_crs}")

    # If usecols specified, filter columns in spatial_signatures_gdf
    if usecols:
        spatial_signatures_gdf = spatial_signatures_gdf[["geometry", "type"] + usecols]
    else:
        spatial_signatures_gdf = spatial_signatures_gdf[["geometry", "type"]]

    # Spatial join for for labelling UPRN with signature type
    labelled_uprn_gdf = uprn_gdf.sjoin(
        spatial_signatures_gdf,
        how="left",
        predicate="intersects",  # include properties intersecting spatial signature cell boundary
    ).drop(columns="index_right")

    # Add potential heat network zone boolean label to original uprn_gdf
    labelled_uprn_gdf["in_potential_hn_zone"] = (
        labelled_uprn_gdf["type"].isin(types) if types else True
    )
    labelled_uprn_gdf["in_potential_hn_zone"] = labelled_uprn_gdf[
        "in_potential_hn_zone"
    ].fillna(False)

    # Return as polars df without geometry
    labelled_uprn_df = pl.from_pandas(labelled_uprn_gdf.drop(columns="geometry"))

    labelled_uprn_df = labelled_uprn_df.rename({"type": "spatial_signature_type"})

    return labelled_uprn_df
