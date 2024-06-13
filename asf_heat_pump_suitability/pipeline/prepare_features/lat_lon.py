import polars as pl
from asf_heat_pump_suitability.getters import get_datasets


def prepare_df_osopen_uprn_latlon() -> pl.DataFrame:
    """
    Args:
        df (pl.DataFrame):

    Returns:
        pl.DataFrame:
    """
    df = get_datasets.get_df_osopen_uprn_latlon()
    # Following line required to convert UPRNs to same format as in EPC
    df = df.with_columns(pl.col("UPRN").cast(pl.Float64).cast(pl.String).alias("UPRN"))

    return df
