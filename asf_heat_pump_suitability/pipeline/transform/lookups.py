import polars as pl


def transform_df_uprn_lookup(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean UPRN national statistics lookup 'PCDS' column data and rename to 'postcode', and extend with 'country' column derived from 'ctry25cd'.

    Args:
        df (pl.DataFrame): UPRN national statistics lookup dataframe containing columns: PCDS and ctry25cd.

    Returns:
        pl.DataFrame: cleaned UPRN lookup.
    """
    return df.with_columns(
        pl.col("PCDS").str.replace(r"\s+", "").alias("postcode"),
        pl.col("ctry25cd").str.slice(0, 1).alias("country"),
    )
