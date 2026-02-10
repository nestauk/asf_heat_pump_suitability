import polars as pl
import geopandas as gpd


def generate_df_features(
    buildings_gdf: gpd.GeoDataFrame, uprns_gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Args:
        buildings_gdf (gpd.GeoDataFrame): building footprints.
        uprns_gdf (gpd.GeoDataFrame): UPRNs  with `property_type_flat` boolean data.
        id_col (str): name of building ID column
    """
    buildings_w_uprns_gdf = buildings_gdf.sjoin(
        uprns_gdf, how="left", predicate="contains"
    )
    uprns_w_buildings_gdf = uprns_gdf.sjoin(
        buildings_gdf, how="left", predicate="within"
    )

    features_dfs = [
        generate_df_building_features(buildings_w_uprns_gdf, id_col),
        generate_df_stacked_uprn_features(uprns_w_buildings_gdf, id_col),
        generate_df_concave_hull_features(uprns_w_buildings_gdf, id_col),
    ]

    features_dfs = pl.align_frames(features_dfs, on=id_col, how="left")

    return pl.concat(features_dfs, how="align_left")


def generate_df_building_features(gdf: gpd.GeoDataFrame, id_col: str) -> pl.DataFrame:
    """
    Args:
        gdf (gpd.GeoDataFrame): building footprints with UPRNs joined to them. UPRNs must have `property_type_flat` boolean data.
        id_col (str): name of building ID column
    """
    gdf["building_area_m2"] = gdf.area
    gdf["building_perimeter"] = gdf.length

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


def generate_df_stacked_uprn_features(
    gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Args:
        gdf (gpd.GeoDataFrame): UPRNs and geometries with building footprints joined to them
        id_col (str): name of building ID column
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


def generate_df_concave_hull_features(
    gdf: gpd.GeoDataFrame, id_col: str
) -> pl.DataFrame:
    """
    Args:
        gdf (gpd.GeoDataFrame): UPRNs and geometries with building footprints joined to them
        id_col (str): name of building ID column
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
