import polars as pl
import logging


def filter_df_epc_building_footprints(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter out EPC rows with no building height data.

    Args:
        df (pl.DataFrame): building height data joined to EPC data

    Returns:
        pl.DataFrame: EPC records with building height data
    """
    n = (df["height"] == -1).sum()
    logging.info(
        f"Dropping {n} rows of building footprint data with no building height information"
    )
    return df.filter(pl.col("height") != -1)


def deduplicate_df_epc_building_footprints(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop all rows with duplicated UPRNs.

    Args:
        df (pl.DataFrame): dataset with `UPRN` column

    Returns:
        pl.DataFrame: dataset with all duplicated UPRNs removed
    """
    n = df["UPRN"].is_duplicated().sum()
    logging.info(f"Dropping {n} rows with duplicated UPRNs")
    return df.unique(subset="UPRN", keep="none")


def extend_df_building_rise(
    df: pl.DataFrame, mps: float, roof_height: float = 2.3
) -> pl.DataFrame:
    """
    Add `building_rise` column to dataframe using `FLAT_STOREY_COUNT` and `height` columns. Buildings are partitioned
    into low- (<=3 storeys), medium- (4-10 storeys) and high-rise (>10 storeys).

    Args:
        df (pl.DataFrame): EPC records with `FLAT_STOREY_COUNT` and building `height` data
        mps (float): meters per storey used to calculate storey counts from building height
        roof_height (float): constant roof height in meters to use in calculation. Default 2.3 which is the national
        regulation minimum ceiling height in the UK.

    Returns:
        pl.DataFrame: EPC data with `building_rise` column
    """
    df = clean_col_flat_storey_count(df)
    df = extend_df_building_storey_count(df, mps, roof_height)

    df = df.with_columns(
        pl.col("calculated_storey_count")
        .fill_null(pl.col("FLAT_STOREY_COUNT"))
        .alias("storey_count")
    ).with_columns(
        pl.when((pl.col("storey_count") > 0) & (pl.col("storey_count") <= 3))
        .then(pl.lit("low-rise"))
        .when((pl.col("storey_count") > 3) & (pl.col("storey_count") <= 10))
        .then(pl.lit("medium-rise"))
        .when(pl.col("storey_count") > 10)
        .then(pl.lit("high-rise"))
        .otherwise(None)
        .alias("building_rise")
    )

    return df


def extend_df_building_storey_count(
    df: pl.DataFrame, mps: float, roof_height: float
) -> pl.DataFrame:
    """
    Calculate storey count of a building by dividing building height by `mps` value.

    Args:
        df (pl.DataFrame): dataset with `height` column
        mps (float): meters per storey used to calculate storey counts from building height
        roof_height (float): constant roof height in meters to use in calculation

    Returns:
        pl.DataFrame: dataframe with `calculated_storey_count` column
    """
    df = df.with_columns(
        ((pl.col("height") - roof_height) / mps)
        .round()
        .alias("calculated_storey_count")
    )

    return df


def clean_col_flat_storey_count(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean `FLAT_STOREY_COUNT` column in EPC data.

    Args:
        df (pl.DataFrame): EPC data with `FLAT_STOREY_COUNT` column

    Returns:
        pl.DataFrame: EPC data with cleaned `FLAT_STOREY_COUNT` column
    """
    logging.warning("Replacing FLAT_STOREY_COUNT values of `20+` with `20`")
    df = df.with_columns(
        pl.col("FLAT_STOREY_COUNT")
        .cast(pl.String)
        .replace(
            "20+", "20"
        )  # This is appropriate here because we are using storeys to determine rise
        .replace("", "unknown")
        .cast(pl.Float64, strict=False)
    ).with_columns(
        pl.when(pl.col("FLAT_STOREY_COUNT") <= 0)
        .then(None)
        .otherwise(pl.col("FLAT_STOREY_COUNT"))
        .alias("FLAT_STOREY_COUNT")
    )

    return df
