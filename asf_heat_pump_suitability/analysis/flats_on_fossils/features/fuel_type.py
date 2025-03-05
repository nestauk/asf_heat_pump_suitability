import polars as pl


def extend_df_fuel_type(df: pl.DataFrame) -> pl.DataFrame:
    """ """
    df = extract_col_main_fuel(df)
    df = extract_col_mainheat_description(df).drop(
        ["fill_fuel_type", "fill_community_heating_system"]
    )
    df = extend_df_fossil_fuel_heating(df)

    return df


def extend_df_fossil_fuel_heating(df: pl.DataFrame) -> pl.DataFrame:
    """ """
    fossil_fuels = ["gas", "oil", "coal", "LNG", "LPG"]

    df = df.with_columns(
        pl.when(pl.col("fuel_type").is_in(fossil_fuels))
        .then(True)
        .when(pl.col("fuel_type").is_not_null())
        .then(False)
        .otherwise(None)
        .alias("fossil_fuel_heating")
    )

    return df


def extract_col_main_fuel(df: pl.DataFrame) -> pl.DataFrame:
    """ """
    # Extract main fuel information into smaller number of unique values
    df = df.with_columns(
        pl.when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("mains gas"))
        .then(pl.lit("gas"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("oil"))
        .then(pl.lit("oil"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("lng"))
        .then(pl.lit("LNG"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("electric"))
        .then(pl.lit("electricity"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("biofuel"))
        .then(pl.lit("biofuel"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("biogas"))
        .then(pl.lit("biogas"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("lpg"))
        .then(pl.lit("LPG"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("b30"))
        .then(pl.lit("B30"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("coal"))
        .then(pl.lit("coal"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("biomass"))
        .then(pl.lit("biomass"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("waste combustion"))
        .then(pl.lit("waste combustion"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("wood"))
        .then(pl.lit("wood"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("anthracite"))
        .then(pl.lit("coal"))
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains("bioethanol"))
        .then(pl.lit("bioethanol"))
        .when(
            pl.col("MAIN_FUEL")
            .str.to_lowercase()
            .str.contains("there is no heating/hot-water system")
        )
        .then(pl.lit("no heating"))
        .otherwise(None)
        .alias("fuel_type"),
        pl.when(
            pl.col("MAIN_FUEL").str.to_lowercase()
            == (r"no heating\/hot-water system or data is from a community network")
        )
        .then(None)
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains(r"\(community\)"))
        .then(True)
        .when(
            pl.col("MAIN_FUEL")
            .str.to_lowercase()
            .str.contains("community heating schemes")
        )
        .then(True)
        .when(pl.col("MAIN_FUEL").str.to_lowercase().str.contains(r"\(not community\)"))
        .then(False)
        .otherwise(None)
        .alias("community_heating_system"),
    )

    return df


def extract_col_mainheat_description(df: pl.DataFrame) -> pl.DataFrame:
    """ """
    # Fill any missing fuel type or community heating system values with MAINHEAT_DESCRIPTION where possible
    df = df.with_columns(
        pl.when(
            pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("mains gas")
        )
        .then(pl.lit("gas"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("lng"))
        .then(pl.lit("LNG"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("oil"))
        .then(pl.lit("oil"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("biofuel"))
        .then(pl.lit("biofuel"))
        .when(
            pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("electric")
        )
        .then(pl.lit("electricity"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("biogas"))
        .then(pl.lit("biogas"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("LPG"))
        .then(pl.lit("LPG"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("b30"))
        .then(pl.lit("B30"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("coal"))
        .then(pl.lit("coal"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("biomass"))
        .then(pl.lit("biomass"))
        .when(
            pl.col("MAINHEAT_DESCRIPTION")
            .str.to_lowercase()
            .str.contains("waste combustion")
        )
        .then(pl.lit("waste combustion"))
        .when(pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("wood"))
        .then(pl.lit("wood"))
        .when(
            pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("anthracite")
        )
        .then(pl.lit("coal"))
        .when(
            pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("bioethanol")
        )
        .then(pl.lit("bioethanol"))
        .when(
            pl.col("MAINHEAT_DESCRIPTION")
            .str.to_lowercase()
            .str.contains("there is no heating/hot-water system")
        )
        .then(pl.lit("no heating"))
        .otherwise(None)
        .alias("fill_fuel_type"),
        pl.when(
            pl.col("MAINHEAT_DESCRIPTION").str.to_lowercase().str.contains("community")
        )
        .then(True)
        .otherwise(None)
        .alias("fill_community_heating_system"),
    ).with_columns(
        pl.col("fuel_type").fill_null(pl.col("fill_fuel_type")),
        pl.col("community_heating_system").fill_null(
            pl.col("fill_community_heating_system")
        ),
    )

    return df
