import polars as pl


def fill_col_missing_garden_size(
    df: pl.DataFrame,
    estimate_col: str = "garden_area_m2",
    avg_col: str = "msoa_avg_outdoor_space_m2",
) -> pl.DataFrame:
    """
    Fill missing garden size estimates with MSOA average outdoor space reported by ONS first, then fill remaining
    missing garden size estimates with MSOA median average calculated from available garden size estimates per MSOA.

    Args:
        df (pl.DataFrame): EPC dataset with estimated garden area, and ONS MSOA average outdoor space columns
        estimate_col (str): name of estimated garden area column with missing values to fill. Default "garden_area_m2".
        avg_col (str): name of ONS MSOA average outdoor space column. Default "msoa_avg_outdoor_space_m2".

    Returns:
        pl.DataFrame: EPC dataset with missing estimated garden area data filled with MSOA averages, and added
        "msoa_median_garden_area_m2" column.
    """
    msoa_median_estimate_df = (
        df.select(["msoa", estimate_col])
        .group_by(["msoa"])
        .median()
        .rename({estimate_col: "msoa_median_garden_area_m2"})
    )

    df = df.join(msoa_median_estimate_df, how="left", on="msoa")

    df = df.with_columns(
        pl.col(estimate_col)
        .fill_null(pl.col(avg_col))
        .fill_null(pl.col("msoa_median_garden_area_m2"))
        .alias(estimate_col),
    )

    return df


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
    houses = ["Detached", "Semi-Detached", "Terraced (including end-terrace)"]

    df = df.with_columns(
        pl.when(pl.col(ptype_col).is_in(houses))
        .then(pl.lit("Houses"))
        .when(pl.col(ptype_col).str.to_lowercase().str.contains("flat"))
        .then(pl.lit("Flats"))
        .otherwise(pl.lit("unknown"))
        .alias("msoa_avg_outdoor_space_property_type")
    )

    return df


def clean_df_nrooms(
    df: pl.DataFrame, nrooms_col: str = "NUMBER_HABITABLE_ROOMS"
) -> pl.DataFrame:
    """
    Clean the EPC number of rooms feature to align with census target data.

    Args:
        df (pl.DataFrame): EPC dataset with number of rooms column
        nrooms_col (str): name of number of rooms column. Default "NUMBER_HABITABLE_ROOMS".

    Returns:
        pl.DataFrame: EPC dataset with new 'nrooms' column with cleaned number of rooms
            (e.g. grouped into a '9+' category)
    """

    df = df.rename({nrooms_col: "nrooms"})
    df = df.with_columns(
        pl.col("nrooms").map_elements(
            lambda x: "9+" if x >= 9 else str(x), return_dtype=pl.String
        )
    )
    df = df.with_columns(
        pl.Series(name="nrooms", values=df["nrooms"].replace(None, "unknown"))
    )

    return df


def extend_df_country_col(df: pl.DataFrame, lsoa_col: str = "lsoa") -> pl.DataFrame:
    """
    Add a new column to a dataframe based on LSOA/DataZone code.

    Args:
        df (pl.DataFrame): dataframe with column containing LSOA / DataZone code
        lsoa_col (str): column containing LSOA / DataZone code

    Returns:
        pl.DataFrame: dataframe with new "country" column
    """
    df = df.with_columns(pl.col(lsoa_col).str.slice(0, 1).alias("country"))
    df = df.with_columns(
        pl.col("country").replace({"E": "England", "S": "Scotland", "W": "Wales"})
    )

    return df
