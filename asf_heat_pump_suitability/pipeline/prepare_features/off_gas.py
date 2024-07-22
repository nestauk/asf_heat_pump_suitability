from asf_heat_pump_suitability.getters.get_datasets import get_df_spa_offgasgrid
import polars as pl


def add_off_gas_feature(df: pl.DataFrame) -> pl.DataFrame:
    """
    This function adds an 'OFF_GAS' column to a DataFrame based on whether 'POSTCODE' is in off_gas_postcodes.

    Parameters:
    df (pl.DataFrame): The EPC DataFrame to which the 'OFF_GAS' column will be added.

    Returns:
    pl.DataFrame: The DataFrame with the added 'OFF_GAS' column.
    """
    off_gas_df = get_df_spa_offgasgrid()
    off_gas_postcodes = off_gas_df["Post Code"].str.replace(" ", "").to_list()
    df = df.with_columns(pl.col("POSTCODE").is_in(off_gas_postcodes).alias("OFF GAS"))
    return df
