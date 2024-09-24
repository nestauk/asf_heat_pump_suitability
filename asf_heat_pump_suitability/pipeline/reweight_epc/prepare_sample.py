import polars as pl
import polars.selectors as cs
from asf_heat_pump_suitability import config


def drop_nulls_feature_cols(df, features):
    """
    Drop rows with null values in any feature column from EPC dataset.

    Args:
        df (pl.DataFrame): EPC dataset
        features (list): column names of features

    Returns:
        pl.DataFrame: EPC dataset where rows with null in any specified feature column are dropped
    """
    df = (
        df.with_columns(  # replace "unknown" with null in string feature columns
            (cs.string() & cs.by_name(features)).replace(
                {"unknown": None}, return_dtype=pl.String
            )
        )
        .with_columns(
            pl.col("build_year").fill_null("unknown")
        )  # replace null with unknown because it's a build_year category
        .drop_nulls(subset=features)
    )
    return df


def add_cols_weighting_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add standardised feature columns to be used for weighting to EPC dataset.

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with standardised feature columns
    """
    df = df.rename({"TENURE": "tenure"})
    df = add_col_property_type(df)
    df = add_col_build_year_1930(df)
    # df = add_col_nrooms(df)  # TODO: collapse into fewer groups (currently 1-9+)
    return df


def add_col_property_type(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `property_type` column to EPC dataset with property type categories corresponding to those from the census.

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with `property_type` column
    """
    terraced = [
        "Mid-Terrace",
        "End-Terrace",
        "Enclosed Mid-Terrace",
        "Enclosed End-Terrace",
    ]

    df = df.with_columns(
        pl.when(
            pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]),
            pl.col("BUILT_FORM") == "Detached",
        )
        .then(pl.lit("Detached"))
        .when(
            pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]),
            pl.col("BUILT_FORM") == "Semi-Detached",
        )
        .then(pl.lit("Semi-detached"))
        .when(
            pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]),
            pl.col("BUILT_FORM").is_in(terraced),
        )
        .then(pl.lit("Terraced (including end-terrace)"))
        .when(pl.col("PROPERTY_TYPE").is_in(["Flat", "Maisonette"]))
        .then(pl.lit("Flat, maisonette or apartment"))
        .when(pl.col("PROPERTY_TYPE").is_in(["Park home"]))
        .then(pl.lit("Caravan or other mobile or temporary structure"))
        .alias("property_type")
    )

    return df


def add_col_nrooms(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `number_of_rooms` column to EPC dataset with categories corresponding to those from the census.

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with `number_of_rooms` column
    """
    return df.with_columns(
        pl.col("NUMBER_HABITABLE_ROOMS")
        .map_elements(lambda x: 9 if x > 9 else x, return_dtype=pl.Float32)
        .cast(pl.Int8)
        .alias("number_of_rooms")
    )


def add_col_build_year_1930(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `build_year` column to EPC dataset indicating whether property construction year is pre- or post-1930 (or unknown).

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with `build_year` column
    """
    return df.with_columns(
        pl.col("CONSTRUCTION_AGE_BAND")
        .replace(config["mapping"]["pre_post_1930_epc"])
        .alias("build_year")
    )
