import logging
import polars as pl
import s3fs
from typing import Optional
from argparse import ArgumentParser
from asf_heat_pump_suitability.pipeline.prepare_features import (
    garden_space_avg,
    lat_lon,
    number_of_households,
    land_area,
    property_density,
    off_gas,
    listed_buildings,
)
from asf_heat_pump_suitability.pipeline.enhance_epc import prepare_epc


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
    - number of households, land area, property density and off gas properties per LSOA

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
    enhanced_epc_df = enhanced_epc_df.join(uprn_latlon_df, how="left", on="UPRN")

    logging.info("Adding number of households data to EPC")
    lsoa_number_of_households_df = (
        number_of_households.prepare_df_num_of_households_ons()
    )
    epc_lsoa_number_of_households_df = lsoa_number_of_households_df.select(
        ["lsoa21", "Number of households 2021"]
    )
    enhanced_epc_df = enhanced_epc_df.join(
        epc_lsoa_number_of_households_df, how="left", on="lsoa21"
    )
    logging.info("Adding land area to EPC")
    lsoa_land_area_df = land_area.prepare_df_land_area_ons()
    epc_lsoa_land_area_df = lsoa_land_area_df.select(
        ["lsoa21", "Land Count (Area in KM2)"]
    )
    enhanced_epc_df = enhanced_epc_df.join(
        epc_lsoa_land_area_df, how="left", on="lsoa21"
    )
    logging.info("Adding property density to EPC")
    enhanced_epc_df = property_density.extend_df_with_property_density(enhanced_epc_df)
    logging.info("Adding off gas grid column to EPC")
    off_gas_postcodes = off_gas.process_off_gas_data()
    enhanced_epc_df = off_gas.add_off_gas_feature(enhanced_epc_df, off_gas_postcodes)

    # Add feature: listed buildings data
    logging.info("Loading listed buildings for England")
    e_listed_buildings_df = listed_buildings.transform_gdf_listed_buildings("England")
    e_listed_buildings_df = listed_buildings.sjoin_df_epc_with_listed_buildings(
        enhanced_epc_df, e_listed_buildings_df
    )

    logging.info("Loading listed buildings for Wales")
    w_listed_buildings_df = listed_buildings.transform_gdf_listed_buildings("Wales")
    w_listed_buildings_df = listed_buildings.sjoin_df_epc_with_listed_buildings(
        enhanced_epc_df, w_listed_buildings_df
    )

    listed_buildings_df = pl.concat(
        [e_listed_buildings_df, w_listed_buildings_df], how="vertical"
    )
    enhanced_epc_df = enhanced_epc_df.join(listed_buildings_df, how="left", on="UPRN")

    # Save to S3
    fs = s3fs.S3FileSystem()
    with fs.open(save_output, mode="wb") as f:
        enhanced_epc_df.write_parquet(f)

    return enhanced_epc_df


if __name__ == "__main__":
    run()
