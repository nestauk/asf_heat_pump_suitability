"""
Assigns a suitable technology and computes feasibility scoring for each cluster.

This file contains functions for:
- preparing the dataframe for feasibility scoring and suitability categorisation;
- categorising/assigning each cluster with the most suitable technology.
- computing the feasibility scores for different tech types and clusters;
"""

# package imports
import polars as pl
import config


def create_df_suitability_categorisation(cluster_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a new column to cluster_df with most suitable low carbon heating technology for each cluster:
    - "individual_ashp": Individual air source heat pump (ASHP)
    - "collective_ashp": Collective purchasing of ASHP
    - "shared_ground_loop": Shared Ground Loop (SGL)
    - "heat_network": Heat Network (HN)

    Args:
        cluster_df (pl.DataFrame): DataFrame containing information about each cluster of properties.

    Returns:
        pl.DataFrame: cluster_df with an additional column for the suitability categorisation.
    """

    most_suitable_tech = (
        pl.when(pl.col("cluster_size") == 1)
        .then(pl.lit("individual_ashp"))
        .when(pl.col("in_heat_network_zone"))
        .then(pl.lit("heat_network"))
        .when(pl.col("city_centre"))
        .then(pl.lit("heat_network"))
        .when((pl.col("cluster_size") > 20) & pl.col("has_outdoor_space"))
        .then(pl.lit("shared_ground_loop"))
        .otherwise(pl.lit("collective_ashp"))
    )

    return cluster_df.with_columns(most_suitable_tech=most_suitable_tech)


def prepare_df_for_feasibility_scoring(
    df: pl.DataFrame,
    anchor_loads_threshold: float = 500,
    city_centre_threshold: float = 500,
) -> pl.DataFrame:
    """
    Prepares the dataframe for feasibility scoring. This includes:
    - Transforming categorical variables into dummy/indicator variables.
    - Creating new binary features based on existing columns and specified thresholds.
    - Scaling the cluster size to be between 0 and 100.
    - Renaming columns for clarity.

    Args:
        df (pl.DataFrame): DataFrame containing the dataset with feature values for each UPRN.
        anchor_loads_threshold (float, optional): threshold for distance to anchor loads in metres. Defaults to 500 m.
        city_centre_threshold (float, optional): threshold for distance to city centre in metres. Defaults to 500 m.

    Returns:
        pl.DataFrame: DataFrame ready for feasibility scoring.
    """

    # Transform predicted_tenure and predicted_property_type into dummies
    df = df.to_dummies("predicted_tenure")
    df = df.rename(
        {
            "predicted_tenure_owner-occupied": "owner_occupied",
            "predicted_tenure_rental (social)": "social_housing",
        }
    )

    df = df.to_dummies("predicted_property_type")
    df = df.rename({"predicted_property_type_Flat, maisonette or apartment": "flats"})

    # Compute on_gas
    df = df.df((~pl.col("use_off_gas")).alias("on_gas"))

    # Compute not_in_listed_building
    df = df.with_columns(
        (~pl.col("in_listed_building")).alias("not_in_listed_building")
    )
    df = df.with_columns(
        (~pl.col("in_conservation_area")).alias("not_in_conservation_area")
    )

    # Create has_outdoor_space from garden_area_m2
    df = df.with_columns((pl.col("garden_area_m2") > 0).alias("has_outdoor_space"))

    # scale cluster size to be between 0 and 100
    df = df.with_columns(
        (
            (pl.col("cluster_size") - pl.col("cluster_size").min())
            / (pl.col("cluster_size").max() - pl.col("cluster_size").min())
            * 100
        ).alias("cluster_size")
    )

    # Creating close_to_anchor_loads from distance_to_anchor_loads and anchor_loads_threhold
    df = df.with_columns(
        (pl.col("distance_to_anchor_loads") <= anchor_loads_threshold).alias(
            "close_to_anchor_loads"
        )
    )
    # Creating close_to_city_centre from distance_to_city_centre and city_centre_threshold
    df = df.with_columns(
        (pl.col("distance_to_city_centre") <= city_centre_threshold).alias(
            "close_to_city_centre"
        )
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
        raise ValueError(
            "There are incorrect or missing keys in the 'weights' dictionary."
        )

    for tech in weights.keys():
        tech_features = set(weights.get(tech).keys())
        if not tech_features.issubset(set(features + ["cluster_size"])):
            raise ValueError(
                f"{tech}: The features you're providing weights for do not exist."
            )

    # Aggregating data by cluster
    cluster_stats = df.group_by("cluster").agg(
        ((pl.col(features).mean()).cast(pl.Float64) * 100).name.prefix("perc_"),
        pl.col("cluster").count().alias("cluster_size"),
    )

    # Create feasibility expressions for all tech types and store as list
    tech_feasibility_scores = [
        calculate_feasibility_expression(tech_specific_weights=weights.get(tech)).alias(
            tech + "_feasibility"
        )
        for tech in expected_tech_types
    ]

    # Add feasibility scores as new columns to cluster_stats
    cluster_stats = cluster_stats.with_columns(tech_feasibility_scores)

    return cluster_stats
