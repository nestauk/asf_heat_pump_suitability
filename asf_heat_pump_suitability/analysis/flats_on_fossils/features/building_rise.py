import polars as pl
import numpy as np
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


def calculate_dict_rise_thresholds(df: pl.DataFrame) -> dict:
    """
    Calculate median meters per storey per building rise type, and thresholds for minimum height of mid-rises and high-rises.

    Args:
        df (pl.DataFrame): EPC flat records with `FLAT_STOREY_COUNT` and building `height` variables

    Returns:
        dict: meters per storey for each building rise type and threshold for min height of mid- and high-rises
    """
    df = df.filter(
        (pl.col("height").is_not_null()) & (pl.col("FLAT_STOREY_COUNT").is_not_null())
    ).with_columns(
        (pl.col("height") / pl.col("FLAT_STOREY_COUNT")).alias("meters_per_storey"),
        pl.when((pl.col("FLAT_STOREY_COUNT") > 0) & (pl.col("FLAT_STOREY_COUNT") <= 3))
        .then(pl.lit("low-rise"))
        .when((pl.col("FLAT_STOREY_COUNT") > 3) & (pl.col("FLAT_STOREY_COUNT") <= 10))
        .then(pl.lit("medium-rise"))
        .when(pl.col("FLAT_STOREY_COUNT") > 10)
        .then(pl.lit("high-rise"))
        .otherwise(None)
        .alias("building_rise"),
    )

    low_rise_uq = df.filter(pl.col("building_rise") == "low-rise")["height"].quantile(
        0.75
    )
    mid_rise_lq = df.filter(pl.col("building_rise") == "medium-rise")[
        "height"
    ].quantile(0.25)

    mid_rise_uq = df.filter(pl.col("building_rise") == "medium-rise")[
        "height"
    ].quantile(0.75)
    high_rise_lq = df.filter(pl.col("building_rise") == "high-rise")["height"].quantile(
        0.25
    )

    median_df = df.group_by("building_rise").agg(pl.col("meters_per_storey").median())
    thresholds = {
        "meters_per_storey": dict(
            zip(median_df["building_rise"], median_df["meters_per_storey"])
        )
    }

    thresholds.update(
        {
            "rise_thresholds": {
                "mid_rise_min": np.mean([low_rise_uq, mid_rise_lq]),
                "high_rise_min": np.mean([mid_rise_uq, high_rise_lq]),
            }
        }
    )

    return thresholds


def extend_df_building_rise(
    df: pl.DataFrame, mid_rise_min: float, high_rise_min: float, mps: dict
) -> pl.DataFrame:
    """
    Add `building_rise` column to dataframe using `FLAT_STOREY_COUNT` and `height` columns. Buildings are partitioned
    into low- (<=3 storeys), medium- (4-10 storeys) and high-rise (>10 storeys).

    Args:
        df (pl.DataFrame): EPC records with `FLAT_STOREY_COUNT` and building `height` data
        mid_rise_min (float): minimum building height of medium rises in meters
        high_rise_min (float): minimum building height of high rises in meters
        mps (dict): meters per storey used to calculate storey counts from building height

    Returns:
        pl.DataFrame: EPC data with `building_rise` column
    """
    df = clean_col_flat_storey_count(df)
    df = (
        df.with_columns(
            # Use flat storey count first
            pl.when(
                (pl.col("FLAT_STOREY_COUNT") > 0) & (pl.col("FLAT_STOREY_COUNT") <= 3)
            )
            .then(pl.lit("low-rise"))
            .when(
                (pl.col("FLAT_STOREY_COUNT") > 3) & (pl.col("FLAT_STOREY_COUNT") <= 10)
            )
            .then(pl.lit("medium-rise"))
            .when(pl.col("FLAT_STOREY_COUNT") > 10)
            .then(pl.lit("high-rise"))
            # Then use building height data
            .when(pl.col("height") >= high_rise_min)
            .then(pl.lit("high-rise"))
            .when(pl.col("height") >= mid_rise_min)
            .then(pl.lit("medium-rise"))
            .when(pl.col("height").is_not_null())
            .then(pl.lit("low-rise"))
            .otherwise(None)
            .alias("building_rise")
        )
        .with_columns(
            pl.col("building_rise")
            .replace(mps)
            .cast(pl.Float64)
            .alias("meters_per_storey")
        )
        .with_columns(
            (pl.col("height") / pl.col("meters_per_storey"))
            .round()
            .alias("derived_storey_count")
        )
        .with_columns(
            pl.col("FLAT_STOREY_COUNT")
            .fill_null(pl.col("derived_storey_count"))
            .alias("storey_count")
        )
    )

    return df


def other_extend_df_building_rise(
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
