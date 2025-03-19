import polars as pl
from asf_heat_pump_suitability.getters import get_target


def transform_df_proportion_census_property_types() -> pl.DataFrame:
    """
    Create dataset of proportions of different property types from the census per LSOA/DZ for England, Scotland, and Wales.

    Returns:
        pl.DataFrame: census counts and proportions of different property types per LSOA/DZ
    """
    df = get_target.transform_df_target_property_type()

    all_property_types = [
        "Detached",
        "Semi-detached",
        "Terraced (including end-terrace)",
        "Flat, maisonette or apartment",
        "Caravan or other mobile or temporary structure",
    ]

    df = (
        df.with_columns(pl.sum_horizontal(all_property_types).alias("total_count"))
        .unpivot(
            index=["lsoa", "total_count"],
            variable_name="property_type",
            value_name="census_count",
        )
        .with_columns(
            (pl.col("census_count") / pl.col("total_count")).alias("census_proportion")
        )
    )

    return df
