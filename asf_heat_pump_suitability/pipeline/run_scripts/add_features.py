import logging
import polars as pl
import s3fs
from typing import Optional
from argparse import ArgumentParser
from asf_heat_pump_suitability.pipeline.prepare_features import (
    garden_space_avg,
    lat_lon,
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

    main(**vars(args))


def main(epc_path: str, save_output: Optional[str] = None) -> pl.DataFrame:
    """
    Enhance EPC dataset with additional features:
    - mean average garden size per MSOA
    - lat/lon per UPRN

    Args
        epc_path (str): S3 URI to EPC dataset with weights and LSOA; MSOA columns
        save_output (str): S3 path to save enhanced EPC dataset to. Optional.

    Returns
        pl.DataFrame: enhanced EPC dataset with additional features
    """
    # Import processed EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    epc_df = pl.read_parquet(epc_path)

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

    # Add feature: conservation area flag
    logging.info("Adding conservation area flag")
    conservation_areas_gdf = get_datasets.load_gdf_historic_england_conservation_areas()

    # Join enhanced datasets together
    enhanced_epc_df = enhanced_epc_df.join(uprn_latlon_df, how="left", on="UPRN")
    epc_gdf = lat_lon.generate_gdf_uprn_coords(enhanced_epc_df)
    # TODO: spatial join between EPC gdf and conservation areas gdf

    # Save to S3
    fs = s3fs.S3FileSystem()
    with fs.open(save_output, mode="wb") as f:
        enhanced_epc_df.write_parquet(f)

    return enhanced_epc_df


if __name__ == "__main__":
    run()
