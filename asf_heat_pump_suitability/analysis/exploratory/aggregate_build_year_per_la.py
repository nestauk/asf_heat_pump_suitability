"""
Aggregate the build year data up to local authority level.
"""

import polars as pl
from asf_heat_pump_suitability.utils import save_utils


if __name__ == "__main__":

    lad_to_lsoa = pl.read_csv(
        "s3://asf-heat-pump-suitability/source_data/2021_vApr2023_ons_lsoa_to_lad_lookup_EW.csv"
    )
    lsoa_build_year = pl.read_csv(
        "s3://asf-heat-pump-suitability/source_data/2015cdrc_dwelling_ages_E_W.csv"
    )

    lsoa_to_lad_dict = dict(zip(lad_to_lsoa["LSOA21CD"], lad_to_lsoa["LAD23CD"]))
    lad_name_dict = dict(zip(lad_to_lsoa["LAD23CD"], lad_to_lsoa["LAD23NM"]))

    lsoa_build_year = lsoa_build_year.with_columns(
        pl.col("AREA_CODE").replace(lsoa_to_lad_dict).alias("LAD23CD")
    )

    lsoa_build_year = lsoa_build_year.with_columns(
        pl.col("LAD23CD").replace(lad_name_dict).alias("LAD23NM")
    )

    pre_columns = ["BP_PRE_1900", "BP_1900_1918", "BP_1919_1929"]

    post_columns = [
        "BP_1930_1939",
        "BP_1945_1954",
        "BP_1955_1964",
        "BP_1965_1972",
        "BP_1973_1982",
        "BP_1983_1992",
        "BP_1993_1999",
        "BP_2000_2009",
        "BP_2010_2015",
    ]

    # Calculate pre- and post-1930 columns
    lsoa_build_year = lsoa_build_year.with_columns(
        [
            pl.sum_horizontal(pre_columns).alias("pre_1930"),
            pl.sum_horizontal(post_columns).alias("post_1930"),
        ]
    ).rename({"BP_UNKNOWN": "unknown", "AREA_CODE": "lsoa"})

    # Aggregate totals per local authority
    la_build_year = lsoa_build_year.group_by(["LAD23CD", "LAD23NM"]).agg(
        pl.col(["pre_1930", "post_1930", "unknown"]).sum()
    )

    # Join local authority totals to LSOA code
    la_build_year = lsoa_build_year.select(["lsoa", "LAD23CD", "LAD23NM"]).join(
        la_build_year, how="left", on=["LAD23CD", "LAD23NM"]
    )

    save_utils.save_parquet_to_s3(
        df=la_build_year,
        path="s3://asf-heat-pump-suitability/source_data_minor_edits/2015cdrc_dwelling_ages_E_W_per_la_02.parquet",
    )
