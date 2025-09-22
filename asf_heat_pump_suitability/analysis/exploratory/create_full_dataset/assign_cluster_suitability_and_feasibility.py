"""
Assigns a suitable technology and computes feasibility scoring for each cluster.

This file contains functions for:
- preparing the dataframe for feasibility scoring and suitability categorisation;
- categorising/assigning each cluster with the most suitable technology.
- computing the feasibility scores for different tech types and clusters;
"""

# package imports
import polars as pl
import geopandas as gpd

# local imports
import config
from asf_heat_pump_suitability.utils.save_utils import save_to_s3


def prepare_df_for_suitability_categorisation(
    df: pl.DataFrame,
    city_centre_oas: set = config.city_centre_oas,
    outdoor_space_threshold: float = config.outdoor_space_threshold,
) -> pl.DataFrame:
    """
    Prepares the dataframe for suitability categorisation. This includes:
    - Identifying clusters close to the city center or in heat network zones
    - Identifying clusters with outdoor space

    Args:
        df (pl.DataFrame): DataFrame containing the dataset with feature values for each UPRN.
        city_centre_oas (set, optional): OAs that are considered part of the city center. Should be in OA21 codes format.
            Defaults to config.city_centre_oas.
        outdoor_space_threshold (float, optional): Threshold for outdoor space in meters squared.
        i.e. if garden_area_m2 > 0, then has_outdoor_space = True

    Returns:
        pl.DataFrame: DataFrame ready for suitability categorisation.
    """
    # Create city_centre column based on whether the UPRN is in the set of city centre OAs
    df = (
        df.with_columns(
            pl.col("oa21")
            .is_in(city_centre_oas)
            .alias("in_city_centre")
            # Transform predicted_tenure and predicted_property_type into dummies
        )
        .to_dummies("predicted_property_type")
        .rename({"predicted_property_type_Flat, maisonette or apartment": "flats"})
    )

    cluster_df = df.group_by("cluster").agg(
        # Cluster size
        pl.col("UPRN").count().alias("cluster_size"),
        # Heat network zone logic: if at least one property in the cluster is in a heat network zone,
        # the whole cluster is considered to be in a heat network zone
        pl.col("in_heat_network_zone").max().alias("in_heat_network_zone"),
        # City centre logic: if at least one property in the cluster is in the city centre,
        # the whole cluster is considered to be close to the city centre
        pl.col("in_city_centre").max().alias("in_city_centre"),
        # Outdoor space logic:
        (
            # Properties other than flats: sum garden_area_m2 for all properties
            pl.col("garden_area_m2")
            .filter(pl.col("flats") != 1)
            .sum()
            .fill_null(0)
        ).alias("total_outdoor_space"),
    )

    # Calculate garden sizes for flats
    flats_garden_sizes = (
        df.group_by("cluster")
        .agg(
            (
                (
                    pl.struct(["building_id", "garden_area_m2"])
                    .filter(pl.col("flats") == 1)
                    .gather(
                        pl.col("building_id").filter(pl.col("flats") == 1).arg_unique()
                    )
                    .struct.field("garden_area_m2")
                )
            )
        )
        .with_columns(
            pl.col("building_id").list.sum().alias("total_outdoor_space_flats")
        )
    )

    # Join the aggregated data together, summing the flats and the not flats for each cluster

    cluster_df = (
        cluster_df.join(flats_garden_sizes, on="cluster", how="left")
        .with_columns(
            (
                pl.col("total_outdoor_space")
                + pl.col("total_outdoor_space_flats").fill_null(0)
            ).alias("total_outdoor_space")
        )
        .drop(["building_id", "total_outdoor_space_flats"])
    )

    cluster_df = cluster_df.with_columns(
        (pl.col("total_outdoor_space") > outdoor_space_threshold).alias(
            "has_outdoor_space"
        )
    )
    return cluster_df


def create_df_suitability_categorisation(
    df: pl.DataFrame, sgl_min_properties: int = config.sgl_min_properties
) -> pl.DataFrame:
    """
    Adds a new column to the dataframe with most suitable low carbon heating technology for each cluster:
    - "individual_ashp": Individual air source heat pump (ASHP)
    - "collective_ashp": Collective purchasing of ASHP
    - "shared_ground_loop": Shared Ground Loop (SGL)
    - "heat_network": Heat Network (HN)

    Args:
        df (pl.DataFrame): DataFrame containing information about each cluster of properties.
        sgl_min_properties (int): The minimum number of properties in a cluster for a SGL to be considered.

    Returns:
        pl.DataFrame: dataframe with an additional column containing the assigned low carbon home heating tech type for each cluster.
    """

    df = df.with_columns(
        pl.when(pl.col("cluster_size") == 1)
        .then(pl.lit("individual_ashp"))
        .when(pl.col("in_heat_network_zone"))
        .then(pl.lit("heat_network"))
        .when(pl.col("in_city_centre"))
        .then(pl.lit("heat_network"))
        .when(
            (pl.col("cluster_size") > sgl_min_properties) & pl.col("has_outdoor_space")
        )
        .then(pl.lit("shared_ground_loop"))
        .otherwise(pl.lit("collective_ashp"))
        .alias("most_suitabile_tech")
    )
    return df


def prepare_df_for_feasibility_scoring(
    df: pl.DataFrame,
    anchor_dist_df: pl.DataFrame,
    features: list = config.features,
    anchor_loads_threshold: float = config.anchor_loads_threshold,
    outdoor_space_threshold: float = config.outdoor_space_threshold,
    city_centre_oas: set = config.city_centre_oas,
) -> pl.DataFrame:
    """
    Prepares the dataframe for feasibility scoring. This includes:
    - Transforming categorical variables into dummy/indicator variables.
    - Creating new binary features based on existing columns and specified thresholds.
    - Renaming columns for clarity.

    Args:
        df (pl.DataFrame): DataFrame containing the dataset with feature values for each UPRN.
        features (list, optional): List of features used in the feasibility scoring.
        anchor_dist_df (pl.DataFrame): DataFrame containing the distance (m) from the centre of each cluster to the nearest anchor load.
        anchor_loads_threshold (float, optional): threshold for distance to anchor loads in metres.
        outdoor_space_threshold (float, optional): Threshold for outdoor space in meters squared.
            i.e. if garden_area_m2 > `outdoor_space_threshold`, then has_outdoor_space = True
        city_centre_oas (set, optional): OAs that are considered part of the city center. Should be in OA21 codes format.

    Returns:
        pl.DataFrame: DataFrame ready for feasibility scoring.
    """

    # Transform predicted_tenure and predicted_property_type into dummies
    df = (
        df.to_dummies("predicted_tenure")
        .to_dummies("predicted_property_type")
        .rename(
            {
                "predicted_tenure_owner-occupied": "owner_occupied",
                "predicted_tenure_rental (social)": "social_housing",
                "predicted_property_type_Flat, maisonette or apartment": "flats",
            }
        )
        .with_columns(
            (~pl.col("use_off_gas")).alias("on_gas"),
            (~pl.col("in_listed_building")).alias("not_in_listed_building"),
            (~pl.col("in_conservation_area")).alias("not_in_conservation_area"),
            (pl.col("garden_area_m2") > outdoor_space_threshold).alias(
                "has_outdoor_space"
            ),
            pl.col("oa21").is_in(city_centre_oas).alias("close_to_city_centre"),
            (pl.col("imd_decile") > 5).alias("imd_decile_above_avg"),
        )
    )

    # Aggregating data by cluster, distance to anchor loads is calculated
    # separately, so don't include this in this step

    df = df.group_by("cluster").agg(
        (
            (pl.col([f for f in features if f != "close_to_anchor_loads"]).mean()).cast(
                pl.Float64
            )
            * 100
        ).name.prefix("perc_"),
        pl.col("UPRN").count().alias("cluster_size"),
    )

    # Creating close_to_anchor_loads from distance_from_anchor_property_m and anchor_loads_threshold

    anchor_dist_df = anchor_dist_df.with_columns(
        (pl.col("distance_from_anchor_property_m") <= anchor_loads_threshold).alias(
            "close_to_anchor_loads"
        )
    ).select(
        pl.col("cluster").cast(pl.Int64),
        (pl.col("close_to_anchor_loads").cast(pl.Float64) * 100).name.prefix("perc_"),
    )

    df = df.join(anchor_dist_df, on="cluster")

    # scale cluster size to be between 0 and 100
    df = df.with_columns(
        (
            (pl.col("cluster_size") - pl.col("cluster_size").min())
            / (pl.col("cluster_size").max() - pl.col("cluster_size").min())
            * 100
        ).alias("cluster_size")
    )

    return df


def calculate_feasibility_expression(tech_specific_weights: dict) -> pl.Expr:
    """
    Generates a Polars expression to calculate a weighted feasibility score.

    Args:
        weights_dict: A dictionary of the format {feature_name: weight}.

    Returns:
        pl.Expr: A Polars expression for the calculated score.
    """
    # Sum of all weights to normalize the score
    total_weight = sum(tech_specific_weights.values())

    # Compute weighted sum i.e. sum(feature_value * weight)
    weighted_cols_sum = pl.sum_horizontal(
        (
            pl.col(f"perc_{feature}") * weight
            if feature != "cluster_size"  # most features will have a "perc_" prefix
            else pl.col("cluster_size")
            * weight  # cluster_size does not have a "perc_" prefix
        )
        for feature, weight in tech_specific_weights.items()
    )

    return weighted_cols_sum / total_weight


def create_df_feasibility_scoring(
    df: pl.DataFrame,
    features: list = config.features,
    expected_tech_types: set = config.expected_tech_types,
    weights: dict = config.weights,
) -> pl.DataFrame:
    """
    Computes the feasibility score for each tech type and cluster in the dataset.

    Tech types can include:
    - Individual Air Source Heat Pump (ASHP)
    - Collective purchasing of ASHP
    - Shared Ground Loop (SGL)
    - Heat Network (HN)

    Args:
        df (pl.DataFrame): DataFrame containing the dataset with feature values for each UPRN.
        features (list, optional): List of features used in the feasibility scoring.
            Defaults to a predefined list.
        expected_tech_types (set, optional): Set of expected technology types for feasibility scoring.
        weights (dict, optional): Dictionary containing the weights for each feature used to compute feasibility for specific tech types.
            in the format {'tech_type_1': {'feature_1': weight_1, 'feature_2': weight_2, ...}, ..., 'tech_type_n': {...}}

    Raises:
        ValueError: if tech types or features provided in weights do not exist.

    Returns:
        pl.DataFrame: DataFrame with aggregated feasibility scores for each cluster.
    """
    if set(weights.keys()) != expected_tech_types:
        incorrect_techs = [w for w in weights.keys() if w not in expected_tech_types]
        missing_techs = [t for t in expected_tech_types if t not in weights.keys()]
        raise ValueError(
            f"There are incorrect or missing keys in the 'weights' dictionary.\n Incorrect techs: {incorrect_techs}\n Missing techs: {missing_techs}"
        )

    for tech in weights.keys():
        input_features = set(weights.get(tech).keys())
        if not input_features.issubset(set(features + ["cluster_size"])):
            do_not_exist = [
                f for f in input_features if f not in set(features + ["cluster_size"])
            ]
            raise ValueError(
                f"{tech}: The features you're providing weights for do not exist:\n{do_not_exist}"
            )

    # Create feasibility expressions for all tech types and store as list
    tech_feasibility_scores = [
        calculate_feasibility_expression(tech_specific_weights=weights.get(tech)).alias(
            tech + "_feasibility"
        )
        for tech in expected_tech_types
    ]

    # Add feasibility scores as new columns to cluster_stats
    df = df.with_columns(tech_feasibility_scores)

    return df


def assign_df_no_cluster_unique_code(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    For any properties not assigned to a cluster, give them a unique negative cluster number (use the UPRN).
    This will mean we can run them through the pipeline as normal, but also distinguish them later too.

    Args:
        df (pl.DataFrame): DataFrame containing the dataset with cluster values for each UPRN.

    Returns:
        pl.DataFrame: DataFrame containing the dataset with cluster values for each UPRN, but where the not clustered properties
        have a unique negative cluster value rather than all being assigned '-1'.
    """

    df = df.with_columns(
        pl.when(pl.col("cluster") == -1)
        .then(pl.col("UPRN") * -1)
        .otherwise(pl.col("cluster"))
        .alias("cluster")
    )

    return df


if __name__ == "__main__":

    plymouth_data = pl.read_parquet(
        "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/results/plymouth_features_selected_with_clusters.parquet"
    )
    plymouth_data = assign_df_no_cluster_unique_code(plymouth_data)

    # The building polygon data per cluster, and the distance to anchor loads from the centre of the cluster
    cluster_polygons = gpd.read_file(
        "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/merged_uprns/per_cluster_merged_polygons.geojson"
    ).to_crs(epsg=4326)

    anchor_dist_df = pl.from_pandas(
        cluster_polygons[["cluster", "distance_from_anchor_property_m"]]
    )

    feasibility_scoring_data = prepare_df_for_feasibility_scoring(
        df=plymouth_data,
        anchor_dist_df=anchor_dist_df,
        features=config.features,
        anchor_loads_threshold=config.anchor_loads_threshold,
        outdoor_space_threshold=config.outdoor_space_threshold,
        city_centre_oas=config.city_centre_oas,
    )

    feasibility_scoring_data = create_df_feasibility_scoring(
        df=feasibility_scoring_data,
        features=config.features,
        expected_tech_types=config.expected_tech_types,
        weights=config.weights,
    )

    suitability_categorisation_data = prepare_df_for_suitability_categorisation(
        df=plymouth_data,
        city_centre_oas=config.city_centre_oas,
        outdoor_space_threshold=config.outdoor_space_threshold,
    )
    suitability_categorisation_data = create_df_suitability_categorisation(
        df=suitability_categorisation_data, sgl_min_properties=config.sgl_min_properties
    )

    # Saving data
    save_to_s3(
        df=feasibility_scoring_data,
        path="s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/plymouth_feasibility_scoring.parquet",
    )
    save_to_s3(
        df=suitability_categorisation_data,
        path="s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/plymouth_suitability_categorisation.parquet",
    )
