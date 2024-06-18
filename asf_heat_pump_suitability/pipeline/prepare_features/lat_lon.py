import polars as pl
from asf_heat_pump_suitability.getters import get_datasets


def transform_df_osopen_uprn_latlon() -> pl.DataFrame:
    """
    Transform UPRN column in raw OS Open UPRN data to match format in EPC.

    Args:
        df (pl.DataFrame): OS Open UPRN dataset

    Returns:
        pl.DataFrame: OS Open UPRN dataset with UPRN column in same format as in EPC
    """
    df = get_datasets.get_df_osopen_uprn_latlon()
    # Following line required to convert UPRNs to same format as in EPC
    df = df.with_columns(pl.col("UPRN").cast(pl.Float64).cast(pl.String).alias("UPRN"))

    return df
