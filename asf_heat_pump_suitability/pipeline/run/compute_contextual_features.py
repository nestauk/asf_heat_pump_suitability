"""
Script to compute contextual information for opportunity areas/clusters, e.g. property type, tenure, EPC rating, etc.

Run:
python asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py --uprns path/to/domestic/UPRNs --epc path/to/deduplicated/EPC --opportunity_areas path/to/opportunity/areas.geojson --save --save_path path/to/save/output.csv
"""

import argparse
from importlib.resources import path
import polars as pl
import geopandas as gpd


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    # TODO this is a placeholder and likely to change as the script develops
    parser.add_argument(
        "--uprns",
        help="Path to domestic UPRN dataset with X and Y coordinates in parquet.",
        type=str,
        required=False,
        default="s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_features.parquet",
    )

    parser.add_argument(
        "--hnz",
        help="Path to domestic UPRN dataset with heat network zone information in parquet.",
        type=str,
        required=False,
        default="s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_hn_zones_city_centres.parquet",
    )
    parser.add_argument(
        "--epc",
        help="Path to deduplicated EPC dataset in parquet.",
        type=str,
        required=False,
        default="s3://asf-daps/lakehouse/2025_Q1/processed/epc/deduplicated/processed_dedupl-0.parquet",
    )

    parser.add_argument(
        "--opportunity_areas",
        help="Path to opportunity areas dataset in kml or geojson format.",
        type=str,
        required=False,
        default="s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_tech_polygons_with_clusterID.geojson",
    )
    parser.add_argument(
        "--save",
        help="Whether to save the output to S3.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_path",
        help="Path to save the output geojson with contextual information per opportunity area.",
        type=str,
        default="s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth/plymouth_cluster_contextual_features.geojson",
    )

    return parser.parse_args()


def process_df_epc_data(epc_df: pl.DataFrame) -> pl.DataFrame:
    """
    Process EPC data to get relevant features for joining to UPRNs.

    Args:
        epc_df (pl.DataFrame): raw EPC dataframe
    Returns:
        pl.DataFrame: processed EPC dataframe with relevant features for joining to UPRNs
    """
    keep_cols = [
        "UPRN",
        "PROPERTY_TYPE",
        "BUILT_FORM",
        "TENURE",
        "CURRENT_ENERGY_RATING",
    ]

    epc_df = (
        raw_epc_df.select(keep_cols)
        .with_columns(
            # Remove any invalid UPRNs (i.e. those IDs which are generated in EPC preprocessing generated from concatenating building ref number and address)
            # These are not true UPRNs that can be used in joins across other datasets
            pl.col("UPRN")
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64)
            .alias("UPRN")
        )
        .drop_nulls(subset="UPRN")
        .with_columns(
            pl.col("PROPERTY_TYPE").cast(pl.String),
            pl.col("BUILT_FORM").cast(pl.String),
        )
        .with_columns(
            # Reassign enclosed terrace categories and set 'flat' as an attachment type
            pl.when(pl.col("BUILT_FORM") == "Enclosed Mid-Terrace")
            .then(pl.lit("Mid-Terrace"))
            .when(pl.col("BUILT_FORM") == "Enclosed End-Terrace")
            .then(pl.lit("End-Terrace"))
            .when(pl.col("BUILT_FORM").str.to_lowercase().is_in(["", "unknown"]))
            .then(None)
            .otherwise(pl.col("BUILT_FORM"))
            .alias("ATTACHMENT")
        )
        .with_columns(
            pl.col("TENURE")
            .str.to_lowercase()
            .replace({"": None, "unknown": None})
            .alias("TENURE")
        )
    )

    return epc_df


def join_df_epc_to_uprns(
    uprns_gdf: gpd.GeoDataFrame, epc_df: pl.DataFrame
) -> gpd.GeoDataFrame:
    """
    Join processed EPC data to UPRN geodataframe.
    Args:
        uprns_gdf (gpd.GeoDataFrame): geodataframe of UPRNs with geometry
        epc_df (pl.DataFrame): processed EPC dataframe with relevant features for joining to UPRNs
    Returns:
        gpd.GeoDataFrame: UPRN geodataframe with EPC features joined
    """

    keep_cols = ["UPRN", "ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]
    uprns_gdf = uprns_gdf.merge(
        epc_df.select(keep_cols).to_pandas(), how="left", on="UPRN"
    )

    # Only fill with Unknown for cols in keep_cols
    uprns_gdf[keep_cols] = uprns_gdf[keep_cols].fillna("Unknown")

    return uprns_gdf


def filter_df_uprns_to_opportunity_areas(
    uprns_gdf: gpd.GeoDataFrame, areas_gdf: gpd.GeoDataFrame
) -> pl.DataFrame:
    """
    Filter UPRN geodataframe to only those which are within opportunity areas.

    Args:
        uprns_gdf (gpd.GeoDataFrame): geodataframe of UPRNs with geometry and EPC features
        areas_gdf (gpd.GeoDataFrame): geodataframe of opportunity areas with geometry and cluster_id
    Returns:
        pl.DataFrame: filtered UPRN dataframe with only UPRNs within opportunity areas
    """
    opportunity_areas_df = pl.from_pandas(
        uprns_gdf.sjoin(
            areas_gdf[["cluster_id", "geometry"]], how="right", predicate="within"
        ).drop(columns=["index_left", "geometry"])
    ).with_columns(
        # Change attachment to flat if it is one
        pl.when(pl.col("property_type_flat"))
        .then(pl.lit("Flat"))
        .otherwise(pl.col("ATTACHMENT"))
        .alias("ATTACHMENT")
    )

    return opportunity_areas_df


def calculate_df_dummy_feature_value_counts_per_opportunity_area(
    opportunity_areas_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate value counts per opportunity area for relevant features.

    Args:
        opportunity_areas_df (pl.DataFrame): dataframe of UPRNs within opportunity areas with relevant features
    Returns:
        pl.DataFrame: dataframe with value counts per opportunity area for relevant features
    """
    keep_cols = ["cluster_id", "UPRN", "ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]
    dummy_cols = ["ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]

    # Get total UPRNs per opportunity area
    totals_df = opportunity_areas_df.group_by("cluster_id").agg(
        pl.col("UPRN").count().alias("n_UPRNs")
    )

    # Get value counts per feature
    opportunity_areas_df = (
        opportunity_areas_df.select(keep_cols)
        .to_dummies(columns=dummy_cols)
        .group_by("cluster_id")
        .agg(pl.all().sum())
        .drop("UPRN")
    )

    # Join total UPRN counts onto value counts
    opportunity_areas_df = totals_df.join(
        opportunity_areas_df, how="left", on="cluster_id"
    )

    return opportunity_areas_df


def create_df_remaining_features_per_opportunity_area(
    opportunity_areas_df: pl.DataFrame,
    uprns_gdf: gpd.GeoDataFrame,
) -> pl.DataFrame:
    """
    Create dataframe with remaining features per opportunity area:
    - Average garden size
    - etc

    Args:
        opportunity_areas_df (pl.DataFrame): dataframe with value counts per opportunity area for relevant features
    Returns:
        pl.DataFrame: dataframe with remaining features per opportunity area
    """

    # Get the cluster_id for each UPRN by spatially joining UPRN geodataframe with opportunity area geodataframe
    uprns_df = pl.from_pandas(
        uprns_gdf.sjoin(
            areas_gdf[["cluster_id", "geometry"]], how="left", predicate="within"
        ).drop(columns=["geometry"])
    )

    # Average contigous outdoor space per opportunity area
    avg_outdoor_space = uprns_df.group_by("cluster_id").agg(
        pl.col("max_contiguous_outdoor_space_area_m2").mean().alias("avg_outdoor_space")
    )

    # Join total UPRN counts onto value counts
    opportunity_areas_df = opportunity_areas_df.join(
        avg_outdoor_space, how="left", on="cluster_id"
    )

    # Create HN zone and city centre flags per opportunity area - if any UPRN within the area is in a HN zone or city centre, then the whole area is flagged as being in a HN zone or city centre
    hnz_city_centre_flags = uprns_df.group_by("cluster_id").agg(
        pl.when(pl.col("in_hn_zone").any())
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("in_hn_zone"),
        pl.when(pl.col("in_city_centre").any())
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("in_city_centre"),
    )

    opportunity_areas_df = opportunity_areas_df.join(
        hnz_city_centre_flags, how="left", on="cluster_id"
    )

    # TODO add remaining features - the section below is temporary
    new_cols = [
        "n_uprns_listed_building",
        "n_uprns_off_gas",
        "near_coast_line",
        "near_anchor_load",
    ]

    opportunity_areas_df = opportunity_areas_df.with_columns(
        [pl.lit("Unknown").alias(col) for col in new_cols]
    )

    return opportunity_areas_df


if __name__ == "__main__":
    import polars as pl
    import geopandas as gpd
    from asf_heat_pump_suitability.utils import save_utils
    from asf_heat_pump_suitability.pipeline.transform import uprns

    args = parse_arguments()

    # Load UPRNs
    print("Loading Plymouth domestic UPRNs...")
    uprns_df = pl.read_parquet(args.uprns)
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df).to_crs(epsg=27700)

    hnz_df = pl.read_parquet(args.hnz)
    hnz_gdf = uprns.generate_gdf_uprn_coords(hnz_df).to_crs(epsg=27700)

    uprns_gdf = uprns_gdf.merge(
        hnz_gdf[["UPRN", "in_hn_zone", "in_city_centre"]], how="left", on="UPRN"
    )

    # Load EPC
    print("Loading deduplicated EPC data...")
    raw_epc_df = pl.read_parquet(args.epc)

    # Load opportunity areas
    print("Loading opportunity areas...")
    areas_gdf = gpd.read_file(args.opportunity_areas).to_crs(epsg=27700)

    print("Processing EPC data...")
    epc_df = process_df_epc_data(raw_epc_df)

    print("Joining EPC data to UPRNs...")
    # Add EPC data to UPRNs
    uprns_gdf = join_df_epc_to_uprns(uprns_gdf, epc_df)

    print("Filtering to opportunity areas...")
    # Filter to UPRNs which are in opportunity areas
    opportunity_areas_df = filter_df_uprns_to_opportunity_areas(uprns_gdf, areas_gdf)

    print("Calculate value counts per feature...")
    # Calculate value counts per opportunity area for relevant features
    opportunity_areas_df = calculate_df_dummy_feature_value_counts_per_opportunity_area(
        opportunity_areas_df
    )

    print("Calculate remaining features per opportunity area...")
    # Calculate remaining features per opportunity area
    opportunity_areas_df = create_df_remaining_features_per_opportunity_area(
        opportunity_areas_df, uprns_gdf
    )

    if args.save:
        opportunity_areas_df = opportunity_areas_df.to_pandas().merge(
            areas_gdf[["cluster_id", "geometry"]],
            how="left",
            on="cluster_id",
        )

        # Saving as EPSG:4326 because we need lat/long for visualisation
        opportunity_areas_df = gpd.GeoDataFrame(
            opportunity_areas_df, geometry="geometry", crs="EPSG:4326"
        )

        opportunity_areas_df.to_file(
            args.save_path,
            driver="GeoJSON",
        )
