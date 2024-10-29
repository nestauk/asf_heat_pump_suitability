"""
Subset the EPC data to only have data from local authorities which are above Liverpool (and including Liverpool)
and in England.

This will work by finding the southern most UPRN per LA, and if this is more northern than the most southern
UPRN for Liverpool then all UPRNs from this LA will be included.
"""

import polars as pl
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.prepare_features import lat_lon, output_areas
from asf_heat_pump_suitability.utils import save_utils

import s3fs

if __name__ == "__main__":
    epc_path = (
        "s3://asf-daps/lakehouse/processed/epc/deduplicated/processed_dedupl-0.parquet"
    )
    epc_df = pl.read_parquet(epc_path, columns=config["usecols"]["epc"])

    # Add lat/long and local authority per UPRN
    uprn_latlon_df = lat_lon.transform_df_osopen_uprn_latlon()
    epc_df = epc_df.join(uprn_latlon_df, how="left", on="UPRN")

    epc_df = output_areas.standardise_col_postcode(epc_df, pcd_col="POSTCODE")
    onspd_df = output_areas.transform_df_ons_pd()
    epc_df = epc_df.join(onspd_df, how="left", on="POSTCODE")

    # Replace `lad_code` from postcode with `lad_code` from geospatial join and postcode
    uprn_lad_df = output_areas.sjoin_df_uprn_lad_code(epc_df)  # Takes a long time!
    epc_df = epc_df.drop("lad_code").join(uprn_lad_df, how="left", on="UPRN")

    # Get the UPRNs which are from LAs North of Liverpool
    # Want to include the entire LA of data (so can't just subset on latitude alone)

    # Find the UPRN with the most southern latitude per LAD
    min_lat_per_lad = epc_df[["lad_code", "LATITUDE"]].group_by("lad_code").min()

    # Liverpool LA code = 'E08000012'

    min_lat_liverpool = (
        min_lat_per_lad.filter(pl.col("lad_code") == "E08000012")
        .select(pl.first("LATITUDE"))
        .item()
    )
    lad_in_north = (
        min_lat_per_lad.filter(pl.col("LATITUDE") >= min_lat_liverpool)["lad_code"]
        .unique()
        .to_list()
    )

    # Subset EPC
    epc_df_north = epc_df.filter(pl.col("lad_code").is_in(lad_in_north))

    # Don't include Scottish UPRNs
    epc_df_north = epc_df_north.filter(pl.col("country_code") != "S")

    # Save
    save_as = "s3://asf-heat-pump-suitability/source_data_minor_edits/northern_england_epc_processed_dedupl-0.parquet"
    save_utils.save_parquet_to_s3(epc_df_north, save_as)
