import polars as pl
from asf_heat_pump_suitability.getters import get_datasets


def join_df_additional_features(epc_df: pl.DataFrame) -> pl.DataFrame:
    """
    Join additional features to EPC dataset: LSOA; MSOA.

    Args
        epc_df (pl.DataFrame): EPC dataset

    Returns
        pl.DataFrame: EPC dataset with additional feature columns
    """
    epc_df = _standardise_df_epc(epc_df)
    onspd_df = get_datasets.get_df_onspd_gb()

    df = epc_df.join(onspd_df, how="left", left_on="POSTCODE", right_on="postcode")

    return df


def _standardise_df_epc(epc_df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare EPC dataset for join with additional features: standardise "POSTCODE" column.

    Args
        epc_df (pl.DataFrame): EPC dataset

    Returns
        pl.DataFrame: EPC dataset with standardised postcode column
    """
    return epc_df.with_columns(
        pl.col("POSTCODE").str.to_uppercase().str.replace(" ", "")
    )
