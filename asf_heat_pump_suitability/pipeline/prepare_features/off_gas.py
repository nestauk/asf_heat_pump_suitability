from typing import List
from asf_heat_pump_suitability.getters.get_datasets import get_df_spa_offgasgrid
import polars as pl


def process_off_gas_data() -> List[str]:
    """
    This function processes the off-gas data by removing spaces from the postcodes and converting them to a list.

    Returns:
    List[str]: The list of processed off-gas postcodes.
    """
    off_gas_df = get_df_spa_offgasgrid()
    off_gas_postcodes = off_gas_df["Post Code"].str.replace(r"\s+", "").to_list()
    return off_gas_postcodes


def add_off_gas_feature(df: pl.DataFrame, off_gas_postcodes: List[str]) -> pl.DataFrame:
    """
    This function adds an 'OFF_GAS' column to a DataFrame based on whether 'POSTCODE' is in off_gas_postcodes.

    Parameters:
    df (pl.DataFrame): The EPC DataFrame to which the 'OFF_GAS' column will be added.
    off_gas_postcodes (List[str]): The list of processed off-gas postcodes.

    Returns:
    pl.DataFrame: The DataFrame with the added 'OFF_GAS' column.
    """
    df = df.with_columns(pl.col("POSTCODE").is_in(off_gas_postcodes).alias("OFF GAS"))
    return df
