import polars as pl
from asf_heat_pump_suitability.getters import get_datasets


def extend_df_with_property_density(enhanced_epc_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add property density feature to the EPC dataset
    Args:
        enhanced_epc_df (pl.DataFrame): EPC dataset with additional features
    Returns:
        pl.DataFrame: EPC dataset with property density feature
    """
    # Fill None values with a default value (e.g., 0)
    enhanced_epc_df = enhanced_epc_df.with_columns(
        [
            pl.col("Number of households 2021")
            .fill_null(0)
            .alias("Number of households 2021"),
            pl.col("Land Count (Area in KM2)")
            .fill_null(0)
            .alias("Land Count (Area in KM2)"),  # Fill with 1 to avoid division by zero
        ]
    )
    # Calculate and add the new column with the property density, handling division by zero
    enhanced_epc_df = enhanced_epc_df.with_columns(
        (
            pl.when(pl.col("Land Count (Area in KM2)") != 0)
            .then(
                pl.col("Number of households 2021") / pl.col("Land Count (Area in KM2)")
            )
            .otherwise(None)
        ).alias("households_per_km2")
    )
    # Changes 0 values back to None
    enhanced_epc_df = replace_zeros_with_none_df(
        enhanced_epc_df, "Number of households 2021"
    )
    enhanced_epc_df = replace_zeros_with_none_df(
        enhanced_epc_df, "Land Count (Area in KM2)"
    )

    return enhanced_epc_df


def replace_zeros_with_none_df(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    This function replaces 0 values in the specified column of a DataFrame with None.

    Parameters:
    df (pl.DataFrame): The DataFrame to modify.
    column_name (str): The name of the column in which to replace 0 values.

    Returns:
    pl.DataFrame: The modified DataFrame.
    """
    df = df.with_columns(
        pl.when(pl.col(column_name) == 0)
        .then(None)
        .otherwise(pl.col(column_name))
        .alias(column_name)
    )

    return df


def generate_df_property_density_s() -> pl.DataFrame:
    """
    Generate dataframe with property density (dwellings per km2) per 2011 Data Zone in Scotland. Long-term empty dwellings
    are excluded from dwelling count to calculate property density.

    Returns:
        pl.DataFrame: properties per km2 per 2011 Data Zone in Scotland
    """
    dz_df = load_transform_df_datazone_area()
    dwellings_df = load_transform_df_n_dwellings_s()
    df = dwellings_df.join(dz_df, how="inner", on="DataZone").with_columns(
        (pl.col("n_dwellings") / pl.col("StdAreaKm2")).alias("households_per_km2")
    )

    return df.select(["DataZone", "households_per_km2"])


def load_transform_df_datazone_area():
    """
    Load and transform dataframe with area (km2) per 2011 Scottish Data Zone.

    Returns:
         pl.DataFrame: area (km2) per 2011 Data Zone in Scotland
    """
    df = get_datasets.load_gdf_scotgov_data_zone_bounds()[["DataZone", "StdAreaKm2"]]

    return pl.from_pandas(df)


def load_transform_df_n_dwellings_s():
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
