import polars as pl
from asf_heat_pump_suitability import config


def add_cols_weighting_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add standardised feature columns to be used for weighting, to EPC dataset.

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with standardised feature columns
    """
    df = add_col_property_type(df)
    df = add_col_nrooms(df)
    return add_col_build_year_1930(df)


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
        .then(pl.lit("Detached whole house or bungalow"))
        .when(
            pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]),
            pl.col("BUILT_FORM") == "Semi-Detached",
        )
        .then(pl.lit("Semi-detached whole house or bungalow"))
        .when(
            pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]),
            pl.col("BUILT_FORM").is_in(terraced),
        )
        .then(pl.lit("Terraced (including end-terrace) whole house or bungalow"))
        .when(pl.col("PROPERTY_TYPE").is_in(["Flat", "Maisonette"]))
        .then(pl.lit("Flat, maisonette or apartment"))
        .when(pl.col("PROPERTY_TYPE").is_in(["Park home"]))
        .then(pl.lit("A caravan or other mobile or temporary structure"))
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
        .map_dict(config["mapping"]["pre_post_1930_epc"])
        .alias("build_year")
    )
