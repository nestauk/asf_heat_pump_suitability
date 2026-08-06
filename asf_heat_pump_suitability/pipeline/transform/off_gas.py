from typing import List

import polars as pl
import geopandas as gpd

from asf_heat_pump_suitability.getters import load_data


def load_transform_list_off_gas_postcodes() -> List[str]:
    """
    Clean off-gas postcodes by removing spaces from them and converting them to a list.

    Returns:
        List[str]: The list of processed off-gas postcodes.
    """
    off_gas_df = load_data.load_df_off_gas_pcds()
    off_gas_postcodes = off_gas_df["Post Code"].str.replace(r"\s+", "").to_list()
    return off_gas_postcodes
