"""
Functions to calculate suitability of different HP technologies.
"""

import polars as pl
from asf_heat_pump_suitability.pipeline.suitability import scoring


def compute_df_avg_score_per_epc(
    df: pl.DataFrame,
    tech_type: str,
    density_threshold: int = 100,
    garden_threshold: int = 10,
    external_space_threshold: int = 2,
):
    """
    Calculate average heat pump suitability score per EPC record for specified tech type.

    Args:
        df: EPC dataset with features for calculating suitability score
        tech_type (str): tech type to calculate suitability scores for, in standard (S) or Nesta (N) view
        density_threshold: minimum property density (households per km2) required for shared ground loop
        garden_threshold: minimum garden size (m2) required for heat pumps
        external_space_threshold: minimum outdoor space (m2) required for heat pumps

    Returns:
        pl.DataFrame: average suitability score for specified tech type
    """
    scores_df = compute_df_total_score_per_epc(
        df, tech_type, density_threshold, garden_threshold, external_space_threshold
    )
    max_scores_df = compute_df_max_score_per_epc(df, tech_type)
    df = scores_df.join(max_scores_df, on="UPRN", how="inner")
    df = df.with_columns(
        (pl.col(f"{tech_type}_score") / pl.col(f"{tech_type}_max_score")).alias(
            f"{tech_type}_avg_score"
        )
    )
    return df.select(["UPRN", f"{tech_type}_avg_score"])


def compute_df_total_score_per_epc(
    df: pl.DataFrame,
    tech_type: str,
    density_threshold: int = 60,
    garden_threshold: int = 10,
    external_space_threshold: int = 2,
    grid_installation_threshold: int = 30,
) -> pl.DataFrame:
    """
    Calculate total heat pump suitability score points per EPC record for specified tech_type.

    Args:
        df: EPC dataset with features for calculating suitability score
        tech_type (str): tech type to calculate suitability scores for, in standard (S) or Nesta (N) view
        density_threshold: minimum property density (households per km2) required for shared ground loop
        garden_threshold: minimum garden size (m2) required for heat pumps
        external_space_threshold: minimum outdoor space (m2) required for heat pumps
        grid_installation_threshold: minimum percentage of properties in LSOA that could have a heat pump installed with current grid capacity

    Returns:
        pl.DataFrame: suitability score for specified tech type
    """
    df = df.with_columns(
        pl.when(pl.col("ruc_two_fold") == "Urban")
        .then(scoring.high_heat_demand_scores.get(tech_type))
        .alias("heat_demand_score"),
        pl.when(pl.col("off_gas"))
        .then(scoring.offgas_scores.get(tech_type))
        .alias("off_gas_score"),
        pl.when(pl.col("households_per_km2") > density_threshold)
        .then(scoring.property_density_scores.get(tech_type))
        .alias("property_density_score"),
        pl.when(pl.col("garden_area_m2") > garden_threshold)
        .then(scoring.garden_size_scores.get(tech_type))
        .alias("garden_size_score"),
        pl.when(pl.col("garden_area_m2") > external_space_threshold)
        .then(scoring.external_space_scores.get(tech_type))
        .alias("external_space_score"),
        pl.when(~pl.col("listed_building"))
        .then(scoring.site_regs_scores.get(tech_type))
        .alias("not_listed_score"),
        pl.when(~pl.col("in_protected_area"))
        .then(scoring.site_regs_scores.get(tech_type))
        .alias("not_in_protected_area_score"),
        pl.when(pl.col("property_type") != "Flat, maisonette or apartment")
        .then(scoring.not_flat_scores.get(tech_type))
        .otherwise(scoring.multiple_props_scores.get(tech_type))
        .alias("property_type_score"),
        pl.when(pl.col("CURRENT_ENERGY_RATING").is_in(["A", "B", "C"]))
        .then(scoring.epc_threshold_scores.get(tech_type))
        .alias("epc_rating_score"),
        pl.when(pl.col("has_anchor_property"))
        .then(scoring.anchor_properties_scores.get(tech_type))
        .alias("anchor_properties_score"),
        pl.when(
            pl.col("heatpump_installation_percentage") >= grid_installation_threshold
        )
        .then(scoring.epc_threshold_scores.get(tech_type))
        .alias("grid_capacity_score"),
    )

    score_cols = [col for col in df.columns if "score" in col]
    df = df.with_columns(pl.sum_horizontal(score_cols).alias(f"{tech_type}_score"))

    return df.select(["UPRN", f"{tech_type}_score"])


def compute_df_max_score_per_epc(df: pl.DataFrame, tech_type: str) -> pl.DataFrame:
    """
    Get max score possible per EPC record. This is calculated by adding together the scores for the tech type for each
    feature in the row that is not null.

    Args:
        df: EPC dataset with features for calculating suitability score
        tech_type (str): tech type to calculate suitability scores for, in standard (S) or Nesta (N) view

    Returns:
        pl.DataFrame: max possible score per EPC row
    """
    df = df.with_columns(
        pl.when(pl.col("ruc_two_fold").is_not_null())
        .then(scoring.high_heat_demand_scores.get(tech_type))
        .otherwise(0)
        .alias("heat_demand_max"),
        pl.when(pl.col("off_gas").is_not_null())
        .then(scoring.offgas_scores.get(tech_type))
        .otherwise(0)
        .alias("off_gas_max"),
        pl.when(pl.col("households_per_km2").is_not_null())
        .then(scoring.property_density_scores.get(tech_type))
        .otherwise(0)
        .alias("property_density_max"),
        pl.when(pl.col("garden_area_m2").is_not_null())
        .then(
            scoring.garden_size_scores.get(tech_type)
            + scoring.external_space_scores.get(tech_type)
        )
        .otherwise(0)
        .alias("garden_size_max"),
        pl.when(pl.col("listed_building").is_not_null())
        .then(scoring.site_regs_scores.get(tech_type))
        .alias("listed_buildings_max"),
        pl.when(pl.col("in_protected_area").is_not_null())
        .then(scoring.site_regs_scores.get(tech_type))
        .alias("in_protected_area_max"),
        pl.when(pl.col("property_type").is_not_null())
        .then(
            # Note: this only works because not_flat and multiple_props scores are mutually exclusive
            (
                scoring.not_flat_scores.get(tech_type)
                + scoring.multiple_props_scores.get(tech_type)
            )
        )
        .otherwise(0)
        .alias("property_type_max"),
        pl.when(pl.col("CURRENT_ENERGY_RATING").is_not_null())
        .then(scoring.epc_threshold_scores.get(tech_type))
        .otherwise(0)
        .alias("epc_rating_max"),
        pl.when(pl.col("has_anchor_properties").is_not_null())
        .then(scoring.anchor_properties_scores.get(tech_type))
        .otherwise(0)
        .alias("anchor_properties_max"),
        pl.when(pl.col("heatpump_installation_percentage").is_not_null())
        .then(scoring.epc_threshold_scores.get(tech_type))
        .otherwise(0)
        .alias("grid_capacity_max"),
    )

    max_cols = [col for col in df.columns if "max" in col]
    df = df.with_columns(pl.sum_horizontal(max_cols).alias(f"{tech_type}_max_score"))

    return df.select(["UPRN", f"{tech_type}_max_score"])


def filter_df_minimum_features(
    df: pl.DataFrame, features: list, threshold: int = 4
) -> pl.DataFrame:
    """
    Calculate number of non-null features for each row of EPC and filter for only rows with number of non-null features
    above specified threshold.

    Args:
        df: EPC dataset with features.
        features: list of features to calculate HP suitability
        threshold: minimum features required to be included. Default 4.

    Returns:
        pl.DataFrame: EPC rows with required number of features
    """
    df = df.with_columns(
        (len(features) - pl.sum_horizontal(pl.col(features).is_null()))
    ).rename(
        {"literal": "n_features"}
    )  # Not sure why this is required, naming with alias directly doesn't work fsr

    df = df.filter(pl.col("n_features") >= threshold)

    return df


def compute_df_weighted_score(df, threshold=0.5):
    """
    Calculate [un]weighted suitability scores per EPC property in a single LSOA. Scores will only be weighted if the
    proportion of EPC properties in the LSOA with weight data is above the specified threshold.

    Args:
        df: EPC dataset filtered to a single LSOA with suitability scores per property for each tech type and with proportional weights
        threshold (float): minimum proportion of properties in LSOA EPC subset with weights. Default 0.5.

    Returns:
        pl.DataFrame: [un]weighted scores per EPC record for one LSOA
    """
    score_cols = [col for col in df.columns if "score" in col]
    df = df.with_columns(
        pl.when(
            (pl.col("proportional_weight").is_not_null().sum() / len(df)) >= threshold
        )
        .then(pl.col("proportional_weight") / pl.col("proportional_weight").sum())
        .otherwise(1)  # Otherwise we use a weight of 1 per row
        .alias("use_weight"),
        pl.when(
            (pl.col("proportional_weight").is_not_null().sum() / len(df)) >= threshold
        )
        .then(True)
        .otherwise(False)
        .alias("scores_weighted"),
    )
    for col in score_cols:
        df = df.with_columns(
            (pl.col(col) * pl.col("use_weight")).alias(f"{col}_weighted")
        )

    return df


def compute_dict_lsoa_suitability_scores(df: pl.DataFrame, lsoa: str) -> dict:
    """
    Calculate average heat pump suitability scores for one LSOA per tech type.

    Args:
        df (pl.DataFrame): LSOA with weighted suitability scores per tech type
        lsoa (str): LSOA code

    Returns:
        dict: suitability scores for one LSOA for each tech type
    """
    scores_dict = {"lsoa": lsoa}
    assert df["scores_weighted"].n_unique() == 1
    score_cols = [col for col in df.columns if "score_weighted" in col]
    for score in score_cols:
        if df["scores_weighted"].unique()[0]:
            scores_dict[score] = df[score].sum()
        else:
            scores_dict[score] = df[score].mean()
    scores_dict["scores_weighted"] = df["scores_weighted"].unique()[0]
    scores_dict["n_properties"] = len(df)

    return scores_dict
