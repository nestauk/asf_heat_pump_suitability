import polars as pl


def extend_df_feature_weight(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add new column containing `feature_weight` per property to weighted EPC data for one LSOA.

    Args:
        df (pl.DataFrame): EPC data filtered to single LSOA with one row per property with `scores_weighted` and `proportional_weight` columns

    Returns:
        pl.DataFrame: EPC data for one LSOA with new `feature_weight` column to use for computing weighted proportions of features
    """
    df = df.with_columns(
        pl.when(pl.col("scores_weighted"))
        .then(pl.col("proportional_weight"))
        .otherwise(1)
        .alias("feature_weight"),
        pl.when(pl.col("scores_weighted"))
        .then(1)
        .otherwise(len(df))
        .alias("total_weight"),
    )

    return df


def aggregate_dict_features_per_lsoa(df: pl.DataFrame) -> dict:
    """
    Compute aggregate proportions of heat pump suitability features for one LSOA.

    Args:
        df (pl.DataFrame): heat pump suitability per property dataset with suitability features for one LSOA

    Returns:
        dict: features aggregated to LSOA level for one LSOA where keys are feature names, and values are LSOA-level values
    """
    features_dict = {
        # LSOA-level features which will be the same value in each row per LSOA
        "property_density_km2": df["households_per_km2"].min(),
        "rural_urban_class": df["ruc_two_fold"].min(),
        "has_anchor_property": df["has_anchor_property"].min(),
        "heatpump_installation_percentage": df[
            "heatpump_installation_percentage"
        ].min(),
        # Property (or postcode)-level features which will be different per row
        "median_garden_estimate_m2": df["garden_area_m2"].median(),
        "proportion_in_conservation_area": (
            df["in_protected_area"] * df["feature_weight"]
        ).sum()
        / df["total_weight"].min(),
        "proportion_listed_building": (
            df["listed_building"] * df["feature_weight"]
        ).sum()
        / df["total_weight"].min(),
        "proportion_epc_c_plus": (df["epc_c_plus"] * df["feature_weight"]).sum()
        / df["total_weight"].min(),
        "proportion_off_gas": (df["off_gas"] * df["feature_weight"]).sum()
        / df["total_weight"].min(),
    }

    return features_dict
