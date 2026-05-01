"""
Script to compute contextual information for clusters, e.g. property type, tenure, EPC rating, etc.

Run:
python asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py --local_authorities LOCAL_AUTHORITIES

LOCAL_AUTHORITIES should be one of the options specified in base.yaml's `constant` section, e.g. `plymouth`, `plymouth_similar_cities`, `sampling_areas`, `greater_manchester_las`.

Add --save to save the output to S3 as a geojson with geometry and contextual features per cluster.
"""

import argparse
import polars as pl
import geopandas as gpd

from asf_heat_pump_suitability import config


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities. See base.yaml's `constant` section for options e.g. `plymouth`, `plymouth_similar_cities`, `sampling_areas`, `greater_manchester_las`.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save",
        help="Whether to save the output to S3.",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def filter_df_uprns_to_clusters(
    uprns_gdf: gpd.GeoDataFrame, clusters_gdf: gpd.GeoDataFrame
) -> pl.DataFrame:
    """
    Filter UPRN geodataframe to only those which are within clusters.

    Args:
        uprns_gdf (gpd.GeoDataFrame): geodataframe of UPRNs with geometry and EPC features
        clusters_gdf (gpd.GeoDataFrame): geodataframe of clusters with geometry and cluster_id
    Returns:
        pl.DataFrame: filtered UPRN dataframe with only UPRNs within clusters
    """
    print("len uprns before filtering:", len(uprns_gdf))
    print("len clusters:", len(clusters_gdf))
    # Get the cluster_id for each UPRN by spatially joining UPRN geodataframe with cluster geodataframe
    uprns_df = pl.from_pandas(
        uprns_gdf.sjoin(
            clusters_gdf[["cluster_id", "geometry"]],
            how="right",
            predicate="within",
        ).drop(columns=["geometry"])
    )
    print("len uprns after filtering to clusters:", len(uprns_df))
    return uprns_df


def extend_df_contextual_features(
    clusters_df: pl.DataFrame,
    uprns_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Extend clusters dataframe with contextual features from UPRN geodataframe, including:
    - property type (including flats)
    - tenure
    - EPC rating
    - Median garden size
    - HN zone flag
    - City centre flag
    - number of listed buildings
    - number of off-gas properties
    - proximity to coastline flag
    - conservation area flag

    Note that "within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load` column is added in the cluster.py, so not included here.

    Args:
        clusters_df (pl.DataFrame): dataframe of clusters with cluster_id
        uprns_df (pl.DataFrame): dataframe of UPRNs with cluster_id and relevant features for aggregation
    Returns:
        pl.DataFrame: dataframe with remaining features per cluster
    """

    dummy_cols = ["ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]
    # Get value counts per feature
    dummy_contextual_feat_df = (
        uprns_df.select(dummy_cols + ["cluster_id"])
        .to_dummies(columns=dummy_cols)
        .group_by("cluster_id")
        .agg(pl.all().sum())
    )

    # Keep only the columns that start with the dummy column prefixes, e.g. "ATTACHMENT_", "TENURE_", "CURRENT_ENERGY_RATING_"
    dummy_cols_to_keep = [
        col
        for col in dummy_contextual_feat_df.columns
        if any(col.startswith(prefix) for prefix in dummy_cols)
    ]
    dummy_contextual_feat_df = dummy_contextual_feat_df.select(
        ["cluster_id"] + dummy_cols_to_keep
    )

    # lower case column names
    dummy_contextual_feat_df = dummy_contextual_feat_df.rename(
        {
            col: col.lower().replace(" ", "_").replace("-", "_")
            for col in dummy_contextual_feat_df.columns
        }
    )

    clusters_df = clusters_df.join(
        dummy_contextual_feat_df, how="left", on="cluster_id"
    )

    contextual_feat_clusters_df = (
        uprns_df.group_by("cluster_id")
        .agg(
            # median_outdoor_space
            pl.col("max_contiguous_outdoor_space_area_m2")
            .median()
            .alias("median_outdoor_space_m2"),
            # in_hn_zone flag
            pl.when(pl.col("in_hn_zone").any())
            .then(True)
            .otherwise(False)
            .alias("in_hn_zone"),
            # in_city_centre flag
            pl.when(pl.col("in_city_centre").any())
            .then(True)
            .otherwise(False)
            .alias("in_city_centre"),
            # n_uprns
            pl.col("UPRN").n_unique().alias("n_UPRNs"),
            # n_uprns_listed_building
            pl.col("in_listed_building").sum().alias("n_uprns_in_listed_building"),
            # n_uprns_off_gas
            pl.col("off_gas").sum().alias("n_uprns_off_gas"),
            # near_coastline flag
            pl.col("within_1500m_coastline").any().alias("within_1500m_coastline"),
            # in_conservation_area flag
            pl.col("in_protected_area").any().alias("in_protected_area"),
        )
        .select(
            [
                "cluster_id",
                "median_outdoor_space_m2",
                "in_hn_zone",
                "in_city_centre",
                "n_UPRNs",
                "n_uprns_in_listed_building",
                "n_uprns_off_gas",
                "within_1500m_coastline",
                "in_protected_area",
            ]
        )
    )

    clusters_df = clusters_df.join(
        contextual_feat_clusters_df, how="left", on="cluster_id"
    )

    # Add percentages used for sorting & filtering in the tool
    # owner occupied, social rented, private rented percentages
    tenure_cols = [
        col for col in dummy_contextual_feat_df.columns if col.startswith("tenure_")
    ]
    clusters_df = clusters_df.with_columns(
        [
            (pl.col(col) / pl.col("n_UPRNs") * 100).alias("perc_" + col)
            for col in tenure_cols
        ]
    )

    return clusters_df


if __name__ == "__main__":
    from asf_heat_pump_suitability.pipeline.transform import uprns
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()
    local_authorities = args.local_authorities

    print(f"Loading {local_authorities} domestic UPRNs...")
    uprns_df = pl.read_parquet(
        config["output"]["dataset"]["residential_uprns_with_features"].format(
            local_authority=local_authorities
        )
    )
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df).to_crs(epsg=27700)

    print("Loading opportunity areas...")
    clusters_gdf = gpd.read_file(
        config["output"]["dataset"]["tech_clusters"].format(
            local_authorities=args.local_authorities, tolerance=5
        ),
    ).to_crs(epsg=27700)

    print("Filtering to clusters...")
    uprns_df = filter_df_uprns_to_clusters(
        uprns_gdf=uprns_gdf, clusters_gdf=clusters_gdf
    )

    print("Calculate remaining features per cluster...")
    clusters_with_contextual_features_df = extend_df_contextual_features(
        clusters_df=pl.from_pandas(
            clusters_gdf.drop(columns="geometry")
        ),  # drop geometry for now and use polars
        uprns_df=uprns_df,
    )

    if args.save:
        # Adding the geometry back to the clusters dataframe
        clusters_with_contextual_features_df = (
            clusters_with_contextual_features_df.to_pandas().merge(
                clusters_gdf[["cluster_id", "geometry"]],
                how="left",
                on="cluster_id",
            )
        )

        clusters_with_contextual_features_gdf = gpd.GeoDataFrame(
            clusters_with_contextual_features_df, geometry="geometry", crs="EPSG:27700"
        )

        save_utils.save_to_s3(
            clusters_with_contextual_features_gdf,
            config["output"]["dataset"]["clusters_tech_contextual_info"].format(
                local_authority=local_authorities
            ),
        )
