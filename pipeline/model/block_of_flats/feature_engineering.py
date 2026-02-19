"""Feature engineering for the block-of-flats random forest classifier.

Generates per-building features from building footprint and UPRN geodata.
"""

import geopandas as gpd
import polars as pl


def generate_df_features(
    buildings_gdf: gpd.GeoDataFrame,
    uprns_gdf: gpd.GeoDataFrame,
    id_col: str,
) -> pl.DataFrame:
    """Generate all features required to train the block-of-flats classifier.

    Args:
        buildings_gdf: Building footprints containing at least one domestic property.
        uprns_gdf: All UPRNs located within the footprints in ``buildings_gdf``
            with a ``property_type_flat`` boolean column.
        id_col: Name of the building ID column.

    Returns:
        pl.DataFrame: All features per building ID.
    """
    buildings_w_uprns_gdf = buildings_gdf.sjoin(uprns_gdf, how="inner", predicate="contains")
    uprns_w_buildings_gdf = uprns_gdf.sjoin(buildings_gdf, how="inner", predicate="within")

    features_dfs = [
        _generate_df_building_features(buildings_w_uprns_gdf, id_col),
        _generate_df_stacked_uprn_features(uprns_w_buildings_gdf, id_col),
        _generate_df_concave_hull_features(uprns_w_buildings_gdf, id_col),
    ]

    features_dfs = pl.align_frames(*features_dfs, on=id_col, how="left")
    return pl.concat(features_dfs, how="align")


def _generate_df_building_features(gdf: gpd.GeoDataFrame, id_col: str) -> pl.DataFrame:
    """Generate per-building footprint features.

    Features: n_UPRNs, n_flats, building_area_m2, building_perimeter_m,
    proportion_flats, UPRNs_per_building_m2.

    Args:
        gdf: Building footprints with UPRNs joined to them.
        id_col: Name of the building ID column.

    Returns:
        pl.DataFrame: Per-building features.
    """
    gdf = gdf.copy()
    gdf["building_area_m2"] = gdf.area
    gdf["building_perimeter_m"] = gdf.length

    df = pl.from_pandas(gdf.drop(columns=["geometry"]))

    return (
        df.group_by(id_col)
        .agg(
            pl.col("UPRN").count().alias("n_UPRNs"),
            pl.col("property_type_flat").sum().alias("n_flats"),
            pl.col("building_area_m2").first().name.keep(),
            pl.col("building_perimeter_m").first().name.keep(),
        )
        .with_columns(
            (pl.col("n_flats") / pl.col("n_UPRNs")).alias("proportion_flats"),
            (pl.col("n_UPRNs") / pl.col("building_area_m2")).alias("UPRNs_per_building_m2"),
        )
    )


def _generate_df_stacked_uprn_features(gdf: gpd.GeoDataFrame, id_col: str) -> pl.DataFrame:
    """Generate stacked-UPRN features per building.

    Features: avg_n_stacked_uprns, std_n_stacked_uprns.

    Args:
        gdf: UPRNs with building footprints joined to them.
        id_col: Name of the building ID column.

    Returns:
        pl.DataFrame: Per-building stacked UPRN features.
    """
    df = pl.from_pandas(gdf.drop(columns="geometry"))
    df = df.with_columns(n_stacked_uprns=pl.col("UPRN").count().over(["X_COORDINATE", "Y_COORDINATE"]))
    return df.group_by(id_col).agg(
        pl.col("n_stacked_uprns").mean().alias("avg_n_stacked_uprns"),
        pl.col("n_stacked_uprns").std().alias("std_n_stacked_uprns"),
    )


def _generate_df_concave_hull_features(gdf: gpd.GeoDataFrame, id_col: str) -> pl.DataFrame:
    """Generate concave-hull features per building.

    Features: concave_hull_area_m2, uprns_per_hull_area_m2, flats_per_hull_area_m2.

    Note: values become infinite when all UPRNs share the same coordinates (area=0).
    These are replaced with -1.

    Args:
        gdf: UPRNs with building footprints joined to them.
        id_col: Name of the building ID column.

    Returns:
        pl.DataFrame: Per-building concave-hull features.
    """
    hull_gdf = gdf.dissolve(id_col).concave_hull().reset_index()
    hull_gdf = hull_gdf.rename(columns={0: "geometry"}).set_geometry("geometry")
    hull_gdf["concave_hull_area_m2"] = hull_gdf.area

    agg_df = (
        pl.from_pandas(gdf.drop(columns=["geometry"]))
        .group_by(id_col)
        .agg(
            pl.col("UPRN").count().alias("n_UPRNs"),
            pl.col("property_type_flat").sum().alias("n_flats"),
        )
        .join(
            pl.from_pandas(hull_gdf.drop(columns="geometry")),
            how="left",
            on=id_col,
        )
        .with_columns(
            (pl.col("n_UPRNs") / pl.col("concave_hull_area_m2")).alias("uprns_per_hull_area_m2"),
            (pl.col("n_flats") / pl.col("concave_hull_area_m2")).alias("flats_per_hull_area_m2"),
        )
        .with_columns(
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

    return agg_df.select([id_col, "concave_hull_area_m2", "uprns_per_hull_area_m2", "flats_per_hull_area_m2"])
