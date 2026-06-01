"""
Script to compute contextual information for clusters including:
- Proportion of attachment types, tenure types, EPC ratings of properties within clusters
- Median outdoor space of properties within clusters
- Whether any properties within clusters are in HN zones, city centres, protected areas, off-gas, within 1500m of coastline
- Number of properties, number of properties in listed buildings and number of properties with solar PV

Run:
python asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py --local_authorities LOCAL_AUTHORITIES

Add --save to save the output to S3 as a geojson with geometry and contextual features per cluster.
"""

import argparse
import polars as pl
import geopandas as gpd

from asf_heat_pump_suitability import config

ANCHOR_LOAD_RADIUS = config["constant"]["anchor_radius"]
COASTLINE_DISTANCE_THRESHOLD_M = config["constant"]["coastline"][
    "distance_from_coastline_threshold_m"
]


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire.",
        type=str,
        nargs="+",
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

    # Get the cluster_id for each UPRN by spatially joining UPRN geodataframe with cluster geodataframe
    uprns_df = pl.from_pandas(
        uprns_gdf.sjoin(
            clusters_gdf[["cluster_id", "geometry"]],
            how="right",
            predicate="within",
        ).drop(columns=["geometry"])
    )

    return uprns_df


def extend_df_contextual_features(
    clusters_df: pl.DataFrame,
    uprns_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Extend clusters dataframe with contextual features by aggregating features at UPRN level to cluster level.
    Contextual features added include:
    - Attachment type proportions
    - tenure type proportions
    - EPC rating proportions
    - Median outdoor space
    - HN zone flag
    - City centre flag
    - number of properties in listed buildings
    - number of off-gas properties
    - proximity to coastline flag (within 1500m)
    - protected area flag
    - within anchor load radius flag

    Args:
        clusters_df (pl.DataFrame): dataframe of clusters with cluster_id and `within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load` feature
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
        .sum()
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

    print(uprns_df[["UPRN", "in_listed_building"]])

    contextual_feat_clusters_df = (
        uprns_df.group_by("cluster_id")
        .agg(
            # n_uprns
            pl.col("UPRN").n_unique().alias("n_UPRNs"),
            # n_uprns_listed_building
            pl.col("in_listed_building").sum().alias("n_uprns_in_listed_building"),
            # n_uprns_missing_listed_building_flag
            pl.col("in_listed_building")
            .is_null()
            .sum()
            .alias("n_uprns_missing_listed_building_flag"),
            # n_uprns_solar_pv
            pl.col("has_solar_pv").sum().alias("n_uprns_solar_pv"),
            # n_uprns_missing_solar_pv_flag
            pl.col("has_solar_pv")
            .is_null()
            .sum()
            .alias("n_uprns_missing_solar_pv_flag"),
            # n_uprns_off_gas
            pl.col("off_gas").sum().alias("n_uprns_off_gas"),
            # n_uprns_missing_off_gas_flag
            pl.col("off_gas").is_null().sum().alias("n_uprns_missing_off_gas_flag"),
            # median estimated energy consumption in 12 months (in kWh/m2)
            pl.col("ENERGY_CONSUMPTION_CURRENT")
            .median()
            .alias("median_estimated_energy_consumption_12_months_kwh_per_m2"),
            # median_outdoor_space
            pl.col("max_contiguous_outdoor_space_area_m2")
            .median()
            .alias("median_outdoor_space_m2"),
            # in_hn_zone flag
            pl.when(pl.col("in_hn_zone").is_null().all())
            .then(pl.lit("Unknown"))
            .when(pl.col("in_hn_zone").any())
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias("in_hn_zone"),
            # in_city_centre flag
            pl.when(pl.col("in_city_centre").is_null().all())
            .then(pl.lit("Unknown"))
            .when(pl.col("in_city_centre").any())
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias("in_city_centre"),
            # near_coastline flag
            pl.when(
                pl.col(f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline")
                .is_null()
                .all()
            )
            .then(pl.lit("Unknown"))
            .when(pl.col(f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline").any())
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias(f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline"),
            # in_protected_area flag
            pl.when(pl.col("in_protected_area").is_null().all())
            .then(pl.lit("Unknown"))
            .when(pl.col("in_protected_area").any())
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias("in_protected_area"),
        )
        .select(
            [
                "cluster_id",
                "n_UPRNs",
                "n_uprns_in_listed_building",
                "n_uprns_missing_listed_building_flag",
                "n_uprns_solar_pv",
                "n_uprns_missing_solar_pv_flag",
                "n_uprns_off_gas",
                "n_uprns_missing_off_gas_flag",
                "median_estimated_energy_consumption_12_months_kwh_per_m2",
                "median_outdoor_space_m2",
                "in_hn_zone",
                "in_city_centre",
                f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline",
                "in_protected_area",
            ]
        )
    )

    clusters_df = clusters_df.join(
        contextual_feat_clusters_df, how="left", on="cluster_id"
    )

    # Switching from booleans to Yes/No/Unknown `within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load`
    clusters_df = clusters_df.with_columns(
        pl.when(pl.col(f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load").is_null())
        .then(pl.lit("Unknown"))
        .when(pl.col(f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load"))
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias(f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load")
    )

    # Add percentages used for sorting & filtering in the tool
    # Tenure cols: owner occupied, social rented, private rented percentages
    tenure_cols = [
        col for col in dummy_contextual_feat_df.columns if col.startswith("tenure_")
    ]
    # Adding percentages of properties with solar PV and off-gas, which are also used for filtering in the tool
    perc_cols = tenure_cols + [
        "n_uprns_solar_pv",
        "n_uprns_missing_solar_pv_flag",
        "n_uprns_off_gas",
        "n_uprns_missing_off_gas_flag",
    ]
    clusters_df = clusters_df.with_columns(
        [
            (pl.col(col) / pl.col("n_UPRNs") * 100).alias(
                "perc_" + col if col in tenure_cols else "perc_" + col.split("n_")[1]
            )
            for col in perc_cols
        ]
    )

    return clusters_df


if __name__ == "__main__":
    from asf_heat_pump_suitability.getters import load_geodata
    from asf_heat_pump_suitability.pipeline.transform import uprns, local_authority
    from asf_heat_pump_suitability.pipeline.cluster import cluster
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()
    local_authorities = args.local_authorities
    tolerance_m = config["constant"]["clustering"]["tolerance_m"]

    local_authority_dict = local_authority.get_dict_la_data(local_authorities)

    print(f"Loading {local_authorities} domestic UPRNs...")
    uprns_df = pl.read_parquet(
        config["output"]["dataset"]["domestic_uprns_with_features"].format(
            local_authority=local_authority_dict["url_slug"]
        )
    )
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=local_authority_dict["grid_squares"]
    )

    print("Loading clusters...")
    # clusters_gdf = gpd.read_parquet(
    #     config["output"]["dataset"]["tech_clusters"].format(
    #         local_authorities=local_authority_dict["url_slug"],
    #         tolerance_m=tolerance_m,
    #     ),
    # ).to_crs(epsg=27700)
    clusters_gdf = gpd.read_parquet("test_clusters.parquet").to_crs(epsg=27700)
    desnz_hn_zones_gdf = clusters_gdf[clusters_gdf["cluster_id"].str.contains("DESNZ")]
    clusters_gdf = clusters_gdf[
        ~clusters_gdf["cluster_id"].isin(desnz_hn_zones_gdf["cluster_id"])
    ]

    print("Join UPRNs to clusters...")
    # TODO move cluster-building mapping to cluster.py
    building_cluster_mapping = (
        cluster.sjoin_gdf_buildings_to_clusters(
            buildings_gdf=buildings_gdf, clusters_gdf=clusters_gdf
        )
        .dropna(subset="cluster_id")
        .set_index("ID")["cluster_id"]
        .to_dict()
    )
    uprns_df = uprns_df.with_columns(
        pl.col("ID").replace_strict(building_cluster_mapping).alias("cluster_id")
    )

    building_desnz_mapping = (
        cluster.sjoin_gdf_buildings_to_clusters(
            buildings_gdf=buildings_gdf, clusters_gdf=desnz_hn_zones_gdf
        )
        .dropna(subset="cluster_id")
        .set_index("ID")["cluster_id"]
        .to_dict()
    )
    desnz_uprns_df = uprns_df.with_columns(
        pl.col("ID")
        .replace_strict(building_desnz_mapping, default=None)
        .alias("cluster_id")
    ).drop_nulls(subset="cluster_id")

    uprns_df = pl.concat([uprns_df, desnz_uprns_df])

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
        # Simplify geometries for smaller file size
        clusters_with_contextual_features_gdf["geometry"] = (
            clusters_with_contextual_features_gdf["geometry"].simplify(
                tolerance=tolerance_m
            )
        )

        save_utils.save_to_s3(
            clusters_with_contextual_features_gdf,
            config["output"]["dataset"]["clusters_tech_contextual_info"].format(
                local_authorities=local_authority_dict["url_slug"],
                tolerance_m=tolerance_m,
            ),
        )
