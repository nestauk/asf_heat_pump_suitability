"""
Functions to label UPRNs within official existing, potential or planned heat network zones.
"""

import geopandas as gpd
import polars as pl

from asf_heat_pump_suitability import config


def extend_df_heat_network_zone_bool(
    df: pl.DataFrame, uprns_gdf: gpd.GeoDataFrame, hn_zone_gdf: gpd.GeoDataFrame
) -> pl.DataFrame:
    uprns_in_hnz = generate_dict_heat_network_zone_uprns(
        uprns_gdf=uprns_gdf, hn_zone_gdf=hn_zone_gdf
    ).keys()
    return df.with_columns(
        pl.when(pl.col("UPRN").is_in(uprns_in_hnz))
        .then(True)
        .otherwise(False)
        .alias("in_hn_zone")
    )


def generate_dict_heat_network_zone_uprns(
    uprns_gdf: gpd.GeoDataFrame,
    hn_zone_gdf: gpd.GeoDataFrame,
) -> dict:
    """
    Label UPRNs that are located within a heat network zone.

    Args:
        uprns_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be labelled
        hn_zone_gdf (gpd.GeoDataFrame): polygons of heat network zones

    Returns:
        dict: UPRNs intersecting with heat network zones and their zone identifiers
    """
    print(f"Identifying residential UPRNs in heat network zones...")
    # CRS checks and reprojection if needed
    target_crs = config["constant"]["target_crs"]

    if uprns_gdf.crs != target_crs:
        uprns_gdf = uprns_gdf.to_crs(target_crs)
        print(f"uprn_gdf reprojected to target CRS: {target_crs}")

    if hn_zone_gdf.crs != target_crs:
        hn_zone_gdf = hn_zone_gdf.to_crs(target_crs)
        print(f"hn_zone_gdf reprojected to target CRS: {target_crs}")

    # Assume first column with `ID` substring is the zone ID column
    id_col = [col for col in hn_zone_gdf.columns if "ID" in col][0]
    print(f"Using Heat Network Zone {id_col} column as ID")

    # Spatial join for labelling UPRNs
    labelled_uprn_gdf = (
        uprns_gdf[["UPRN", "geometry"]]
        .sjoin(
            hn_zone_gdf[[id_col, "geometry"]],
            how="inner",
            predicate="intersects",  # include properties intersecting heat network zone boundary
        )
        .drop(columns="index_right")
        .rename({id_col: "HNZoneID"})
    )

    return labelled_uprn_gdf.set_index("UPRN").to_dict()["HNZoneID"]
