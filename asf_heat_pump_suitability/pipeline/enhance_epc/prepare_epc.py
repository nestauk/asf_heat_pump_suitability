import polars as pl


def add_col_msoa_avg_outdoor_space_property_type(
    df: pl.DataFrame, ptype_col: str = "property_type"
) -> pl.DataFrame:
    """
    Add column mapping EPC property type to property types defined in ONS garden size dataset in preparation for
    joining mean average garden size.

    Args:
        df (pl.DataFrame): EPC dataset with property type column
        ptype_col (str): name of property type column. Default "property_type".

    Returns:
          pl.DataFrame: EPC dataset with secondary property type mapped from ONS garden size dataset
    """
    df = df.with_columns(
        pl.when(pl.col(ptype_col).str.to_lowercase().str.contains("house"))
        .then(pl.lit("Houses"))
        .when(pl.col(ptype_col).str.to_lowercase().str.contains("flat"))
        .then(pl.lit("Flats"))
        .otherwise(pl.lit(ptype_col))
        .alias("msoa_avg_outdoor_space_property_type")
    )

    return df
