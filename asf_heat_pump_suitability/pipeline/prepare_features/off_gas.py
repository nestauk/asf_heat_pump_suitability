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
    This function adds an 'off_gas' column to a DataFrame based on whether 'POSTCODE' is in off_gas_postcodes.

    Args:
        df (pl.DataFrame): EPC dataset with postcode column
        off_gas_postcodes (List[str]): The list of processed off-gas postcodes.

    Returns:
        pl.DataFrame: EPC dataframe with the added `off_gas` column.
    """
    df = df.with_columns(pl.col("POSTCODE").is_in(off_gas_postcodes).alias("off_gas"))
    return df
