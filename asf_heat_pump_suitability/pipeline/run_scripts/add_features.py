"""
Enhance EPC dataset with additional features:
- mean average garden size per MSOA
- lat/lon per UPRN
- Historic England conservation area flag
"""

import logging
import polars as pl
import pandas as pd
import s3fs
from typing import Optional
from argparse import ArgumentParser
from asf_heat_pump_suitability.pipeline.prepare_features import (
    conservation_areas,
    garden_space_avg,
    lat_lon,
    output_areas,
)
from asf_heat_pump_suitability.pipeline.enhance_epc import prepare_epc
from asf_heat_pump_suitability.getters import get_datasets


def run():
    """
    Create ArgumentParser and passes arguments to `main()` and runs `main()`.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--epc_path", help="S3 URI to EPC dataset", type=str, required=True
    )

    parser.add_argument(
        "--save_output",
        help="S3 path to save enhanced EPC dataset to",
        type=str,
        default=None,
        required=False,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    _args = run()

    # Import processed EPC
    logging.info(f"Loading EPC file from path: {_args.epc_path}")
    epc_df = pl.read_parquet(_args.epc_path)

    # Join enhancing features to EPC dataset
    # Add feature: garden space avg
    logging.info("Adding average garden size per MSOA to EPC")
    garden_space_avg_msoa_df = garden_space_avg.generate_df_garden_space_avg()
    epc_df = prepare_epc.add_col_msoa_avg_outdoor_space_property_type(epc_df)
    enhanced_epc_df = epc_df.join(
        garden_space_avg_msoa_df,
        how="left",
        left_on=["msoa", "msoa_avg_outdoor_space_property_type"],
        right_on=["MSOA code", "msoa_avg_outdoor_space_property_type"],
    )

    # Add feature: lat/long
    logging.info("Adding lat/lon data to EPC")
    uprn_latlon_df = lat_lon.transform_df_osopen_uprn_latlon()
    enhanced_epc_df = enhanced_epc_df.join(uprn_latlon_df, how="left", on="UPRN")

    # Add feature: conservation area flag
    logging.info("Adding conservation area flag")

    # Join LAD code to EPC
    lad_df = get_datasets.get_df_ons_pd(columns=["pcd", "oslaua"]).rename(
        mapping={"oslaua": "lad_code"}
    )
    lad_df = output_areas.standardise_col_postcode(lad_df, pcd_col="pcd").drop(
        columns=["pcd"]
    )
    enhanced_epc_df = enhanced_epc_df.join(lad_df, how="left", on="POSTCODE")
    # Convert BNG x, y coordinates to point geometries
    enhanced_epc_gdf = lat_lon.generate_gdf_uprn_coords(enhanced_epc_df)

    # Load conservation areas England and identify EPC UPRNs within or on boundaries of conservation areas
    conservation_areas_gdf = (
        conservation_areas.transform_gdf_conservation_areas_england()
    )
    enhanced_epc_df = enhanced_epc_gdf.sjoin(
        conservation_areas_gdf, how="left", predicate="intersects"
    ).drop(columns=["index_right", "geometry"])
    enhanced_epc_df["in_conservation_area"] = (
        enhanced_epc_df["in_conservation_area"].fillna(False).astype(bool)
    )
    # Drop duplicate UPRNs introduced in cases where UPRN matched to multiple conservation areas
    enhanced_epc_df = enhanced_epc_df.drop_duplicates(subset="UPRN")

    # Load conservation areas by LAD to identify LADs with missing conservation area data and join to EPC
    lad_conservation_areas_df = (
        conservation_areas.generate_gdf_conservation_areas_england_lad(
            ladcd_col="LAD23CD"
        )
    )
    enhanced_epc_df = pd.merge(
        enhanced_epc_df,
        lad_conservation_areas_df,
        how="left",
        left_on="lad_code",
        right_on="LAD23CD",
    )

    # Save to S3
    fs = s3fs.S3FileSystem()
    with fs.open(_args.save_output, mode="wb") as f:
        enhanced_epc_df.write_parquet(f)
