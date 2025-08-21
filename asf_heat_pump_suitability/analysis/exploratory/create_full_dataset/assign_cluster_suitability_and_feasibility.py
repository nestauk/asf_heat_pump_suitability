"""
Assigns a suitable technology and computes feasibility scoring for each cluster.

This file contains functions for:
- categorising/assigning each cluster with the most suitable technology scheme.
- computing the feasibility scores for different heating schemes and clusters;
"""

# package imports
import polars as pl


def categorise_cluster_suitability(
    cluster_size: int,
    hn_area: bool,
    city_centre: bool,
    has_outdoor_space: bool,
) -> str:
    """
    Categorises the suitability of a cluster based on its size and characteristics.


    Args:
        cluster_size (int): The number of UPRNs in the cluster.
        hn_area (bool): Indicates if the cluster is in a heat network area.
        city_centre (bool): Indicates if the cluster is close to the city centre.
        has_outdoor_space (bool): Indicates if the cluster has outdoor space.

    Returns:
        str: A string representing the suitability category of the cluster.
    """

    if cluster_size == 1:
        return "individual_ashp"
    elif hn_area:  # if the cluster is in a heat network area
        return "heat_network"
    else:  # not in a heat network area
        if city_centre:
            return "heat_network"
        else:
            if cluster_size > 20:
                if has_outdoor_space:
                    return "shared_ground_loop"
                else:
                    return "collective_ashp"
            else:
                return "collective_ashp"


def create_suitability_categorisation_df(cluster_df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates a new column with the suitability categorisation for each cluster.

    Args:
        cluster_df (pl.DataFrame): DataFrame containing information about each cluster of properties.

    Returns:
        pl.DataFrame: cluster_df with an additional column for the suitability categorisation.
    """

    cluster_df = cluster_df.with_columns(
        pl.struct(["cluster_size", "hn_area", "city_centre", "has_outdoor_space"])
        .map_elements(
            lambda row: categorise_cluster_suitability(
                row["cluster_size"],
                row["hn_area"],
                row["city_center"],
                row["has_outdoor_space"],
            )
        )
        .alias("suitable_scheme")
    )

    return cluster_df


def create_feasibility_scoring_df(
    df: pl.DataFrame,
    weights: dict,
    cols_to_aggregate: list = None,
    anchor_loads_threshold: float = 500,
    city_centre_threshold: float = 500,
) -> pl.DataFrame:
    """
    Computes the feasibility score for each scheme and cluster in the dataset.

    Schemes include:
    - Individual Air Source Heat Pump (ASHP)
    - Collective purchasing of ASHP
    - Shared Ground Loop (SGL)
    - Heat Network (HN)

    Args:
        df (pl.DataFrame): DataFrame containing the dataset with feature values for each UPRN.
        weights (dict): Dictionary containing the weights for each scheme and feature used in feasibility scoring.
        cols_to_aggregate (list, optional): List of columns to aggregate. If None, defaults to a predefined list.
        anchor_loads_threshold (float, optional): default threhold for distance to anchor loads in meteres. Defaults to 500 m.
        city_centre_threshold (float, optional): default threhold for distance to city center in meteres. Defaults to 500 m.

    Returns:
        pl.DataFrame: DataFrame with aggregated feasibility scores for each cluster.
    """

    if cols_to_aggregate is None:
        # Creating a column for not in conservation area
        df = df.with_columns((~pl.col("in_cons_area")).alias("not_in_cons_area"))

        # Creating close_to_anchor_loads from distance_to_anchor_loads and anchor_loads_threhold
        df = df.with_columns(
            (pl.col("distance_to_anchor_loads") <= anchor_loads_threshold).alias(
                "close_to_anchor_loads"
            )
        )
        # Creating close_to_city_centre from distance_to_city_centre and city_centre_threshold
        df = df.with_columns(
            (pl.col("distance_to_city_centre") <= anchor_loads_threshold).alias(
                "close_to_city_centre"
            )
        )

        # Columns to compute percentages for feasibility scoring
        cols_to_aggregate = [
            "owner_occupied",
            "high_income_decile",
            "on_gas",
            "not_listed",
            "not_in_cons_area",
            "social_housing",
            "flats",
            "on_communal_heating",
            "has_outdoor_space",
            "in_listed_buildings",
            "in_cons_area",
            "in_hn",
            "close_to_anchor_loads",
            "close_to_city_centre",
        ]

    # Aggregating data by cluster
    cluster_stats = df.groupby("cluster").agg(
        pl.col(cols_to_aggregate).mean().name.prefix("perc_"),
        pl.col("cluster").count().alias("cluster_size"),
    )

    # Convert proportions to percentages for columns with prefix "perc_"
    cluster_stats = cluster_stats.with_columns(
        (pl.col("^perc_.*$").cast(pl.Float64) * 100).name.keep()
    )

    # Extracting weights for each feasibility scoring category
    individual_ashp_feasibility_weights = weights.get("individual_ashp_feasibility")
    collective_ashp_feasibility_weights = weights.get("collective_ashp_feasibility")
    sgl_feasibility_weights = weights.get("sgl_feasibility")
    hn_feasibility_weights = weights.get("hn_feasibility")

    # Compute feasibility scores using the percentages and the weights
    cluster_stats = cluster_stats.with_columns(
        # Cluster identifier
        pl.col("cluster").alias("cluster"),
        # Number of UPRNs in the cluster
        pl.col("cluster_size").alias("cluster_size"),
        # Feasibility scores for each category
        (
            sum(
                pl.col(f"perc_{key}") * weight
                for key, weight in individual_ashp_feasibility_weights.items()
            )
            / sum(individual_ashp_feasibility_weights.values())
        ).alias("individual_ashp_feasibility"),
        (
            sum(
                pl.col(f"perc_{key}") * weight
                for key, weight in collective_ashp_feasibility_weights.items()
            )
            / sum(collective_ashp_feasibility_weights.values())
        ).alias("collective_ashp_feasibility"),
        (
            sum(
                pl.col(f"perc_{key}") * weight
                for key, weight in sgl_feasibility_weights.items()
            )
            / sum(sgl_feasibility_weights.values())
        ).alias("sgl_feasibility"),
        (
            sum(
                pl.col(f"perc_{key}") * weight
                for key, weight in hn_feasibility_weights.items()
            )
            / sum(hn_feasibility_weights.values())
        ).alias("hn_feasibility"),
    )

    return cluster_stats
