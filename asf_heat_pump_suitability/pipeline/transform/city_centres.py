"""
Functions to label UPRNs within city centre areas.
"""

import logging

import geopandas as gpd
import polars as pl

from asf_heat_pump_suitability import config

logger = logging.getLogger(__name__)

CITY_CENTRE_TYPES = [  # TODO: confirm types with scaling
    "Hyper concentrated urbanity",
    "Concentrated urbanity",
    "Metropolitan urbanity",
    "Regional urbanity",
    "Local urbanity",
    "Dense urban neighbourhoods",
]


def label_gdf_city_centre_spatial_signatures_uprns(
    uprn_gdf: gpd.GeoDataFrame,
    spatial_signatures_gdf: gpd.GeoDataFrame,
    types: list = CITY_CENTRE_TYPES,
) -> pl.DataFrame:
    """
    Labels UPRNs that are located within a city centre based on its matched Spatial Signature type.

    Args:
        uprn_gdf (gpd.GeoDataFrame): UPRNs with point geometries to be labelled
        spatial_signatures_gdf (gpd.GeoDataFrame): polygons of spatial signatures
        types (list, optional): spatial signature types assumed to be representative of city centre areas. Defaults to a list containing:
            - "Hyper concentrated urbanity"
            - "Concentrated urbanity"
            - "Metropolitan urbanity"
            - "Regional urbanity"
            - "Local urbanity"
            - "Dense urban neighbourhoods"

    Returns:
        pl.DataFrame: input UPRNs labelled with spatial signature identifiers and booleans indicating that they are located within
            a city centre spatial signature type
    """

    # CRS checks and reprojection if needed
    target_crs = config["constant"]["target_crs"]

    if uprn_gdf.crs != target_crs:
        uprn_gdf = uprn_gdf.to_crs(target_crs)
        logger.info(f"uprn_gdf reprojected to target CRS: {target_crs}")

    if spatial_signatures_gdf.crs != target_crs:
        spatial_signatures_gdf = spatial_signatures_gdf.to_crs(target_crs)
        logger.info(f"spatial_signatures_gdf reprojected to target CRS: {target_crs}")

    # Spatial join for labelling UPRN with signature type
    labelled_uprn_gdf = uprn_gdf.sjoin(
        spatial_signatures_gdf[["geometry", "type"]],
        how="left",
        predicate="intersects",  # include properties intersecting spatial signature cell boundary
    ).drop(columns="index_right")

    # Add city centre boolean label
    labelled_uprn_gdf["in_city_centre"] = labelled_uprn_gdf["type"].isin(types)

    # Combine multiple matches into a single row per UPRN
    uprn_columns = [col for col in uprn_gdf.columns if col != "geometry"]
    labelled_uprn_gdf = (
        labelled_uprn_gdf.groupby("UPRN", as_index=False)
        .agg(
            {
                **dict.fromkeys(uprn_columns, "first"),  # keep original UPRN columns
                "geometry": "first",  # keep the point geometry
                "type": list,  # combine types into a list
                "in_city_centre": sum,  # sums >0 indicate UPRN in a city centre signature
            }
            # Convert city centre to boolean label
        )
        .astype({"in_city_centre": bool})
    )

    # UPRNs with no spatial signature match are ultimately classified as not in city centre
    # TODO: consider if unassigned UPRNs should be handled differently in the future
    labelled_uprn_gdf["in_city_centre"] = labelled_uprn_gdf["in_city_centre"].fillna(False)

    # Return as polars df without geometry
    labelled_uprn_df = pl.from_pandas(labelled_uprn_gdf.drop(columns="geometry")).rename(
        {"type": "spatial_signature_types"}
    )

    return labelled_uprn_df
