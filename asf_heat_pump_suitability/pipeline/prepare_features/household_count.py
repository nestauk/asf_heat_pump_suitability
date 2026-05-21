import polars as pl
from asf_heat_pump_suitability.getters import get_datasets


def load_transform_df_n_households() -> pl.DataFrame:
    """
    Load and process number of households per LSOA/DataZone in England, Scotland, and Wales.

    Returns
        pl.DataFrame: number of households per LSOA/DataZone
    """
    ew_df = load_transform_df_n_households_ew()
    s_df = load_transform_df_n_dwellings_s().rename(
        {"DataZone": "lsoa", "n_dwellings": "households_count"}
    )

    return pl.concat([ew_df, s_df])


def load_transform_df_n_households_ew() -> pl.DataFrame:
    """
    Load and process ONS number of households per LSOA in England and Wales.

    Returns
        pl.DataFrame: number of households per LSOA
    """
    df = (
        get_datasets.get_df_ons_number_of_households()
        .rename(
            {
                "mnemonic": "lsoa",
                "2021": "households_count",
            }
        )
        .select(["lsoa", "households_count"])
    )
    return df


def load_transform_df_n_dwellings_s() -> pl.DataFrame:
    """
    Load and transform dataframe with number of dwellings per 2011 Scottish Data Zone. Number of dwellings excludes
    long-term empty dwellings.

    Returns:
        pl.DataFrame: number of dwellings per 2011 Scottish Data Zone in Scotland
    """
    df = get_datasets.load_df_nrs_dwellings()
    df = (
        df.with_columns(
            (
                pl.col("Total number of dwellings")
                - pl.col("Long-term empty dwellings \r\n[Note 6]")
            ).alias("n_dwellings")
        )
        .select(["Data Zone code", "n_dwellings"])
        .rename({"Data Zone code": "DataZone"})
    )

    return df
