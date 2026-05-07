"""
Functions to process coastline data.
"""

import geopandas as gpd
import polars as pl


def extend_df_near_coastline_bool(
    features_df: pl.DataFrame,
    uprns_gdf: gpd.GeoDataFrame,
    coast_gdf: gpd.GeoDataFrame,
    distance_threshold_m: int = 1500,
    simplify_tolerance_m: int = 150,
) -> pl.DataFrame:
    """
    Extend dataframe with boolean column indicating whether each UPRN is within the specified distance of the coastline.

    Args:
        features_df (pl.DataFrame): Dataframe containing UPRNs to be labelled with near coastline boolean flag.
        uprns_gdf (gpd.GeoDataFrame): Geodataframe containing UPRNs and their geometries.
        coast_gdf (gpd.GeoDataFrame): Geodataframe containing GB coastline boundaries as a single dissolved geometry.
        distance_threshold_m (int, optional): Distance threshold in metres to define 'near coastline'. Defaults to 1500.
        simplify_tolerance_m (int, optional): Tolerance in metres for simplifying the coastline geometry. Defaults to 150.

    Returns:
        pl.DataFrame: Input dataframe extended with boolean column indicating whether each UPRN is within the specified distance of the coastline.
    """

    # Simplify coastline boundaries by simplify_tolerance_m and buffer by distance_threshold_m to create a 'near coastline' area
    coast_gdf["simplified_geometry"] = coast_gdf.geometry.boundary.simplify(
        tolerance=simplify_tolerance_m
    ).buffer(distance_threshold_m)
    coast_gdf.set_geometry("simplified_geometry", inplace=True)
    coast_gdf[f"within_{distance_threshold_m}m_coastline"] = True

    uprns_gdf = uprns_gdf.sjoin(
        coast_gdf[[f"within_{distance_threshold_m}m_coastline", "simplified_geometry"]],
        how="left",
        predicate="within",
    )

    features_df = features_df.join(
        pl.from_pandas(
            uprns_gdf[["UPRN", f"within_{distance_threshold_m}m_coastline"]]
        ),
        how="left",
        on="UPRN",
    ).with_columns(pl.col(f"within_{distance_threshold_m}m_coastline").fill_null(False))

    return features_df
