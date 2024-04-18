import polars as pl
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline import prepare_features


def join_df_additional_features(epc_df: pl.DataFrame) -> pl.DataFrame:
    """
    Join additional features to EPC dataset: LSOA; MSOA.

    Args
        epc_df (pl.DataFrame): EPC dataset

    Returns
        pl.DataFrame: EPC dataset with additional feature columns
    """
    epc_df = prepare_features.standardise_col_postcode(epc_df, pcd_col="POSTCODE")
    onspd_df = prepare_features.prepare_df_ons_pd()

    df = epc_df.join(onspd_df, how="left", on="POSTCODE")

    return df
