"""
Cluster technology suitability and feasibility scoring.

This file contains functions for:
-  computing the feasibility scores for different heating schemes and clusters;
-  categorising clusters with the most suitable technology scheme.
"""

# package imports
import polars as pl


def create_feasibility_scoring_df(
    df: pl.DataFrame, weights: dict, cols_to_aggregate: list = None
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

    Returns:
        pl.DataFrame: DataFrame with aggregated feasibility scores for each cluster.
    """

    if cols_to_aggregate is None:
        # Creating a column for not in conservation area
        df = df.with_columns((~pl.col("in_cons_area")).alias("not_in_cons_area"))

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
