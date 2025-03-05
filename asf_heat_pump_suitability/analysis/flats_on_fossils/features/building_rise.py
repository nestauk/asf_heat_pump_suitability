import polars as pl


def clean_col_flat_storey_count(df: pl.DataFrame) -> pl.DataFrame:
    """ """
    df = df.with_columns(
        pl.col("FLAT_STOREY_COUNT")
        .cast(pl.String)
        .replace("", "unknown")
        .cast(pl.Float64, strict=False)
    )

    return df


def extend_df_building_rise(df: pl.DataFrame) -> pl.DataFrame:
    """ """
    df = df.with_columns(
        pl.when((pl.col("FLAT_STOREY_COUNT") < 7) & (pl.col("FLAT_STOREY_COUNT") > 0))
        .then(pl.lit("low-rise"))
        .when(pl.col("FLAT_STOREY_COUNT") >= 7)
        .then(pl.lit("high-rise"))
        .otherwise(None)
        .alias("building_rise")
    )

    return df
