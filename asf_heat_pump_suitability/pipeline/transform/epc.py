"""Functions to clean and process features in the EPC register data."""

import polars as pl
from typing import List, Optional


def extend_df_epc_features(
    df: pl.DataFrame, epc_df: pl.DataFrame, columns: Optional[List[str]]
) -> pl.DataFrame:
    """
    Join EPC-derived features to UPRNs. Features added are:
    - `ATTACHMENT` which represents building attachment type (including flats)
    - `TENURE`
    - `CURRENT_ENERGY_RATING` which represents the EPC rating of the property

    Args:
        df (pl.DataFrame): UPRN data with one row per UPRN containing `property_type_flat` column to join new features to..
        epc_df (pl.DataFrame): preprocessed and deduplicated EPC data from asf-daps containing `UPRN`, `BUILT_FORM`, and `TENURE`
        columns.
        columns (List[str]): list of EPC columns to keep, optional. If not set, uses all columns in `epc_df`.

    Returns:
        pl.DataFrame: UPRN data with EPC-derived features where available
    """
    if columns:
        columns += ["UPRN", "BUILT_FORM", "TENURE"]
        epc_df = epc_df.select(set(columns))
    epc_df = clean_extend_df_epc(epc_df)

    return (
        df.join(epc_df, how="left", on="UPRN")
        .with_columns(
            # Change attachment to flat if it is one
            pl.when(pl.col("property_type_flat"))
            .then(pl.lit("Flat"))
            .otherwise(pl.col("ATTACHMENT"))
            .alias("ATTACHMENT")
        )
        .drop("BUILT_FORM")
    )


def clean_extend_df_epc(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean preprocessed EPC dataframe and add new `ATTACHMENT` column to create EPC-derived features for clusters.

    Args:
        df (pl.DataFrame): preprocessed EPC dataframe containing `UPRN`, `BUILT_FORM`, and `TENURE` columns.

    Returns:
        pl.DataFrame: cleaned EPC dataframe with features for clusters.
    """
    df = retain_df_valid_uprns(df, drop=True)
    df = extend_df_attachment_type(df)
    df = clean_df_tenure(df)
    if "PHOTO_SUPPLY" in df.columns and "SOLAR_WATER_HEATING_FLAG" in df.columns:
        df = extend_df_solar_pv_flag(df)
    return df


def retain_df_valid_uprns(df: pl.DataFrame, drop: bool) -> pl.DataFrame:
    """
    Process UPRNs to identify valid UPRNs. Convert any invalid UPRNs (i.e. those IDs which are generated in EPC
    preprocessing generated from concatenating building reference number and address) to null. Valid UPRNs are true
    UPRNs that can be used in joins across other datasets.

    Args:
        df (pl.DataFrame): preprocessed EPC dataframe with `UPRN` column.
        drop (bool): if True, drop rows with invalid UPRNs.

    Returns:
        pl.DataFrame: EPC dataframe with valid UPRNs
    """
    if drop:
        print("Dropping invalid UPRNs from EPC data...")
        return df.with_columns(
            pl.col("UPRN").cast(pl.Float64, strict=False).cast(pl.Int64).alias("UPRN")
        ).drop_nulls(subset="UPRN")

    else:
        return df.with_columns(
            pl.col("UPRN").cast(pl.Float64, strict=False).cast(pl.Int64).alias("UPRN")
        )


def extend_df_attachment_type(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add new ATTACHMENT column to EPC data containing processed building attachment type data. Replaces enclosed mid- and
    end-terrace built forms with mid- and end-terrace, respectively.

    Args:
        df (pl.DataFrame): EPC dataset with `BUILT_FORM` column.

    Returns:
        pl.DataFrame: EPC data with new `ATTACHMENT` column.
    """
    return df.with_columns(
        pl.col("BUILT_FORM").cast(pl.String),
    ).with_columns(
        # Reassign enclosed terrace categories
        pl.when(pl.col("BUILT_FORM") == "Enclosed Mid-Terrace")
        .then(pl.lit("Mid-Terrace"))
        .when(pl.col("BUILT_FORM") == "Enclosed End-Terrace")
        .then(pl.lit("End-Terrace"))
        .when(pl.col("BUILT_FORM").str.to_lowercase().is_in(["", "unknown"]))
        .then(None)
        .otherwise(pl.col("BUILT_FORM"))
        .alias("ATTACHMENT")
    )


def clean_df_tenure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean EPC tenure data.

    Args:
        df (pl.DataFrame): EPC dataset with `TENURE` column.

    Returns:
        pl.DataFrame: EPC data with cleaned `TENURE` column.
    """
    return df.with_columns(
        pl.col("TENURE")
        .str.to_lowercase()
        .replace({"": None, "unknown": None})
        .alias("TENURE")
    )


def extend_df_solar_pv_flag(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create new `has_solar_pv` column indicating presence of photovoltaic panels on a property based on `PHOTO_SUPPLY` column.

    Args:
        df (pl.DataFrame): EPC dataset with `PHOTO_SUPPLY` and `SOLAR_WATER_HEATING_FLAG` columns.

    Returns:
        pl.DataFrame: EPC data with new `has_solar_pv` column.
    """
    has_pv = (pl.col("PHOTO_SUPPLY") > 0) | (
        pl.col("SOLAR_WATER_HEATING_FLAG") == "True"
    )
    no_pv = (pl.col("PHOTO_SUPPLY") == 0) | (
        pl.col("SOLAR_WATER_HEATING_FLAG") == "False"
    )
    return df.with_columns(
        pl.when(has_pv)
        .then(True)
        .when(no_pv)
        .then(False)
        .otherwise(None)
        .alias("has_solar_pv")
    ).drop(["PHOTO_SUPPLY", "SOLAR_WATER_HEATING_FLAG"])
