"""
Functions to generate features from building footprint and UPRN geodata per building for the random forest binary
classifier model which classifies buildings into blocks of flats or not.
"""

import polars as pl
import geopandas as gpd

from asf_heat_pump_suitability.pipeline.transform import building_footprints


def generate_df_features(
    buildings_gdf: gpd.GeoDataFrame, uprns_gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Generate all features required for block of flats random forest binary classifier.

    Args:
        buildings_gdf (gpd.GeoDataFrame): building footprints.
        uprns_gdf (gpd.GeoDataFrame): UPRNs  with `property_type_flat` boolean data.
        id_col (str): name of building ID column

    Return:
        pl.DataFrame: all features for each building ID
    """
    print("Generating features required for block of flats classifier...")
    buildings_w_uprns_gdf = buildings_gdf.sjoin(
        uprns_gdf, how="inner", predicate="intersects"
    ).dropna(subset=id_col)
    uprns_w_buildings_gdf = uprns_gdf.sjoin(
        buildings_gdf, how="inner", predicate="intersects"
    ).dropna(subset="UPRN")

    features_dfs = [
        _generate_df_building_features(buildings_w_uprns_gdf, id_col),
        _generate_df_stacked_uprn_features(uprns_w_buildings_gdf, id_col),
        _generate_df_concave_hull_features(uprns_w_buildings_gdf, id_col),
    ]

    features_dfs = pl.align_frames(*features_dfs, on=id_col, how="left").with_columns(
        (pl.col("concave_hull_area_m2") / pl.col("building_area_m2")).alias(
            "hull_to_building_area_ratio"
        )
    )

    return pl.concat(features_dfs, how="align")


def _generate_df_building_features(gdf: gpd.GeoDataFrame, id_col: str) -> pl.DataFrame:
    """
    Generate select features from building footprints with UPRNs joined to them:
    - n_UPRNs (per building)
    - n_flats (per building)
    - building_area_m2
    - building_perimeter_m
    - proportion_flats (proportion of UPRNs which are flats)
    - UPRNs_per_building_m2 (UPRN density per m2 of building footprint)

    Args:
        gdf (gpd.GeoDataFrame): building footprints with UPRNs joined to them. UPRNs must have `property_type_flat` boolean data.
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: select features per building footprint
    """
    gdf["building_area_m2"] = gdf.area
    gdf["building_perimeter_m"] = gdf.length

    df = pl.from_pandas(gdf.drop(columns=["geometry"]))

    # Aggregate data per building
    agg_building_df = (
        df.group_by(id_col)
        .agg(
            pl.col("UPRN").count().alias("n_UPRNs"),
            pl.col("property_type_flat").sum().alias("n_flats"),
            pl.col("building_area_m2").first().name.keep(),
            pl.col("building_perimeter_m").first().name.keep(),
        )
        .with_columns(
            (pl.col("n_flats") / pl.col("n_UPRNs")).alias("proportion_flats"),
            (pl.col("n_UPRNs") / pl.col("building_area_m2")).alias(
                "UPRNs_per_building_m2"
            ),
        )
    )

    return agg_building_df


def _generate_df_stacked_uprn_features(
    gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Generate select features from UPRNs with building footprints joined to them:
    - avg_n_stacked_uprns (the average number of UPRNs sharing the same coordinates per building)
    - std_n_stacked_uprns (the standard deviation of the number of UPRNs sharing the same coordinates per building)

    Args:
        gdf (gpd.GeoDataFrame): UPRNs and geometries with building footprints joined to them
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: select features per building footprint
    """
    # Get count of UPRNs at each X and Y coordinates to get the count of UPRNs which share an exact location
    df = pl.from_pandas(gdf.drop(columns="geometry"))
    df = df.with_columns(
        # Count of stacked UPRNs per coordinate
        n_stacked_uprns=pl.col("UPRN")
        .count()
        .over(["X_COORDINATE", "Y_COORDINATE"])
    )

    # Group by building and get the average and STD of UPRNs sharing the same coordinates
    df = df.group_by(id_col).agg(
        pl.col("n_stacked_uprns").mean().alias("avg_n_stacked_uprns"),
        pl.col("n_stacked_uprns").std().alias("std_n_stacked_uprns"),
    )

    return df


def _generate_df_concave_hull_features(
    gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Generate select features from UPRNs with building footprints joined to them:
    - concave_hull_area_m2 (the area (m2) of the concave hull of the point geometries of all UPRNs per building)
    - uprns_per_hull_area_m2 (the number of UPRNs per concave hull area per building)
    - flats_per_hull_area_m2 (the number of flats per concave hull area per building)

    Note: UPRNs or flats per hull area can be infinite if all UPRNs/flats share the same coordinates (i.e. area = 0m2).
    In these cases, the uprns_ or flats_per_hull_area_m2 is changed to -1.

    Args:
        gdf (gpd.GeoDataFrame): UPRNs and geometries with building footprints joined to them
        id_col (str): name of building ID column

    Returns:
        pl.DataFrame: select features per building footprint
    """
    # Create concave hull feature to represent spatial distribution of UPRNs in each building
    hull_gdf = gdf.dissolve(id_col).concave_hull().reset_index()
    hull_gdf = hull_gdf.rename(columns={0: "geometry"}).set_geometry("geometry")
    hull_gdf["concave_hull_area_m2"] = hull_gdf.area

    # Aggregate data per building
    agg_building_df = (
        pl.from_pandas(gdf.drop(columns=["geometry"]))
        .group_by(id_col)
        .agg(
            pl.col("UPRN").count().alias("n_UPRNs"),
            pl.col("property_type_flat").sum().alias("n_flats"),
        )
        .join(
            # Join building features with concave hull feature
            pl.from_pandas(hull_gdf.drop(columns="geometry")),
            how="left",
            on=id_col,
        )
        .with_columns(
            # Calculate additional features from the concave hull area
            (pl.col("n_UPRNs") / pl.col("concave_hull_area_m2")).alias(
                "uprns_per_hull_area_m2"
            ),
            (pl.col("n_flats") / pl.col("concave_hull_area_m2")).alias(
                "flats_per_hull_area_m2"
            ),
        )
        .with_columns(
            # UPRNs or flats per hull area can be infinite if all UPRNs/flats share the same coordinates (i.e. area = 0m2)
            # We change this to -1 for the model
            pl.when(pl.col("uprns_per_hull_area_m2").is_infinite())
            .then(-1)
            .otherwise(pl.col("uprns_per_hull_area_m2"))
            .alias("uprns_per_hull_area_m2"),
            pl.when(pl.col("flats_per_hull_area_m2").is_infinite())
            .then(-1)
            .otherwise(pl.col("flats_per_hull_area_m2"))
            .alias("flats_per_hull_area_m2"),
        )
    )

    keep_cols = [
        id_col,
        "concave_hull_area_m2",
        "uprns_per_hull_area_m2",
        "flats_per_hull_area_m2",
    ]

    return agg_building_df.select(keep_cols)


def _generate_df_building_sections_features(
    uprns_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame
) -> pl.DataFrame:
    """ """
    building_units_gdf = building_footprints.generate_gdf_building_sections(
        uprns_gdf=uprns_gdf, buildings_gdf=buildings_gdf
    )
    building_units_gdf["building_unit_area_m2"] = building_units_gdf.area
    building_units_gdf["building_unit_perimeter_m2"] = building_units_gdf.length

    return (
        pl.from_pandas(building_units_gdf.drop(columns="geometry"))
        .group_by("ID")
        .agg(
            pl.col("representative_UPRN").count().alias("n_building_units"),
            pl.col("building_unit_area_m2").mean().alias("avg_building_unit_area_m2"),
            pl.col("building_unit_perimeter_m2")
            .mean()
            .alias("avg_building_unit_perimeter_m2"),
        )
    )
