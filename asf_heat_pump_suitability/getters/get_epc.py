"""Getters for EPC (Energy Performance Certificate) data."""

import logging

import polars as pl

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


def load_set_valid_epc_uprns(epc_type: str) -> set:
    """Load set of valid EPC UPRNs from either commercial or domestic EPC registers.

    Args:
        epc_type (str): {"commercial", "domestic"} the type of EPC to load valid UPRNs from

    Returns:
        set: valid UPRNs from specified EPC dataset
    """
    print(f"Loading UPRNs from {epc_type} EPC register...")
    df = base_getters.load_df_from_s3(config["inputs"]["epc"][epc_type], columns="UPRN")
    before = len(df)
    df = df.with_columns(
        # Remove any invalid UPRNs (i.e. those IDs generated in EPC preprocessing from
        # concatenating building ref number and address). These are not true UPRNs that
        # can be used in joins across other datasets.
        pl.col("UPRN").cast(pl.Float64, strict=False).cast(pl.Int64).alias("UPRN")
    ).drop_nulls()
    logging.info(
        f"{before - len(df)} invalid UPRNs dropped from {epc_type} EPC register. {len(df)} valid UPRNs remaining"
    )
    return set(df["UPRN"])
