import polars as pl


def extend_df_central_heating_information(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `fuel_type` and `community_heating` columns to EPC dataset using data in `MAIN_FUEL` and `MAINHEAT_DESCRIPTION`
    columns. `fuel_type` contains central heating fuel type information, and `community_heating` contains boolean
    values indicating whether the property is on a communal heating system or not.

    Args:
        df (pl.DataFrame): EPC dataset with `MAIN_FUEL` and `MAINHEAT_DESCRIPTION` columns

    Returns:
        pl.DataFrame: EPC dataset with `fuel_type` and `community_heating` columns
    """
    df = extend_df_fuel_type(df, epc_col="MAIN_FUEL", name="fuel_type")
    df = extend_df_fuel_type(df, epc_col="MAINHEAT_DESCRIPTION", name="fill_fuel_type")
    df = extend_df_communal_heating(df, epc_col="MAIN_FUEL", name="community_heating")
    df = extend_df_communal_heating(
        df, "MAINHEAT_DESCRIPTION", name="fill_community_heating"
    )

    # Fill missing fuel type values with hot water fuel type and mains gas flag information
    df = extend_df_hot_water_fuel_type(df, name="fill_fuel_type_2")
    df = df.with_columns(
        pl.col("MAINS_GAS_FLAG")
        .replace("Y", "gas", default=None)
        .alias("fill_fuel_type_3")
    )

    # Fill main fuel type and community heating columns
    df = df.with_columns(
        pl.col("fuel_type")
        .fill_null(pl.col("fill_fuel_type"))
        .fill_null(pl.col("fill_fuel_type_2"))
        .fill_null(pl.col("fill_fuel_type_3")),
        pl.col("community_heating").fill_null(pl.col("fill_community_heating")),
    ).drop(["fill_fuel_type", "fill_community_heating"])

    df = extend_df_fossil_fuel_heating(df)

    return df


def extend_df_fossil_fuel_heating(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `fossil_fuel_heating` column to EPC dataset using derived `fuel_type` column. `fossil_fuel_heating` contains
    boolean values to indicate whether the central heating uses fossil fuels or not.

    Args:
        df (pl.DataFrame): EPC dataset with derived `fuel_type` column

    Returns:
        pl.DataFrame: EPC dataset with `fossil_fuel_heating` column
    """
    fossil_fuels = ["gas", "oil", "coal", "LNG", "LPG", "B30"]

    df = df.with_columns(
        pl.when(pl.col("fuel_type").is_in(fossil_fuels))
        .then(True)
        .when(pl.col("fuel_type").is_not_null())
        .then(False)
        .otherwise(None)
        .alias("fossil_fuel_heating")
    )

    return df


def extend_df_fuel_type(df: pl.DataFrame, epc_col: str, name: str) -> pl.DataFrame:
    """
    Process a string type EPC column containing information about central heating fuel to add a column containing fuel
    type information.

    Args:
        df (pl.DataFrame): EPC dataset
        epc_col (str): name of string type column in EPC data containing central heating fuel information
        name (str): name of new column containing central heating fuel information

    Returns:
        pl.DataFrame: EPC dataset with new column containing central heating fuel type
    """
    # Extract main fuel information into smaller number of unique values
    df = df.with_columns(
        # FOSSIL FUELS
        pl.when(pl.col(epc_col).str.to_lowercase().str.contains("mains gas"))
        .then(pl.lit("gas"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("lng"))
        .then(pl.lit("LNG"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("lpg"))
        .then(pl.lit("LPG"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("b30"))
        .then(pl.lit("B30"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("coal"))
        .then(pl.lit("coal"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("anthracite"))
        .then(pl.lit("coal"))
        # RENEWABLES
        .when(pl.col(epc_col).str.to_lowercase().str.contains("electric"))
        .then(pl.lit("electricity"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("heat pump"))
        .then(pl.lit("electricity"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("biofuel"))
        .then(pl.lit("biofuel"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("biogas"))
        .then(pl.lit("biogas"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("biomass"))
        .then(pl.lit("biomass"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("waste"))
        .then(pl.lit("waste"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("wood"))
        .then(pl.lit("wood"))
        .when(pl.col(epc_col).str.to_lowercase().str.contains("bioethanol"))
        .then(pl.lit("bioethanol"))
        # Oil comes last because it's in the word 'boiler'
        .when(pl.col(epc_col).str.to_lowercase().str.contains("oil"))
        .then(pl.lit("oil"))
        # NO HEATING
        .when(
            pl.col(epc_col)
            .str.to_lowercase()
            .str.contains("there is no heating/hot-water system")
        )
        .then(pl.lit("no heating"))
        .otherwise(None)
        .alias(name)
    )

    return df


def extend_df_communal_heating(
    df: pl.DataFrame, epc_col: str, name: str
) -> pl.DataFrame:
    """
    Process a string type EPC column containing information about central heating type to add a boolean column
    to indicate whether a property uses communal heating systems or not.

    Args:
        df (pl.DataFrame): EPC dataset
        epc_col (str): name of string type column in EPC data containing central heating information
        name (str): name of new column containing communal heating boolean values

    Returns:
        pl.DataFrame: EPC dataset with new column containing communal heating boolean values
    """
    df = df.with_columns(
        pl.when(
            pl.col(epc_col).str.to_lowercase()
            == (r"no heating\/hot-water system or data is from a community network")
        )
        .then(None)
        .when(pl.col(epc_col).str.to_lowercase().str.contains(r"\(not community\)"))
        .then(False)
        .when(pl.col(epc_col).str.to_lowercase().str.contains("community"))
        .then(True)
        .otherwise(None)
        .alias(name)
    )

    return df


def extend_df_hot_water_fuel_type(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """
    Add new column to dataframe with fuel type from EPC `HOTWATER_DESCRIPTION` column.

    Args:
        df (pl.DataFrame): dataset with `HOTWATER_DESCRIPTION` column.
        name (str): name of new column

    Returns:
        pl.DataFrame: dataframe with fuel type of hot water feature
    """
    df = df.with_columns(
        pl.when(
            pl.col("HOTWATER_DESCRIPTION").str.to_lowercase().str.contains("recovery")
        )
        .then(pl.lit("waste"))
        .when(
            pl.col("HOTWATER_DESCRIPTION").str.to_lowercase().str.contains("electric")
        )
        .then(pl.lit("electricity"))
        .when(pl.col("HOTWATER_DESCRIPTION").str.to_lowercase().str.contains("gas"))
        .then(pl.lit("gas"))
        .when(
            pl.col("HOTWATER_DESCRIPTION").str.to_lowercase().str.contains("heat pump")
        )
        .then(pl.lit("electricity"))
        .otherwise(None)
        .alias(name)
    )

    return df
