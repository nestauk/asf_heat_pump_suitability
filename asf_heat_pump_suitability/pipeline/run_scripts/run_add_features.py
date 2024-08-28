"""
Enhance EPC dataset with additional features:
- mean average garden size per MSOA
- lat/lon per UPRN
- number of households, land area, property density and off gas properties per LSOA
- England and Wales building conservation area flag
"""

import logging
import polars as pl
import s3fs
from argparse import ArgumentParser
from datetime import datetime
from asf_heat_pump_suitability.pipeline.prepare_features import (
    conservation_areas,
    epc,
    garden_space_avg,
    lat_lon,
    output_areas,
    number_of_households,
    land_area,
    property_density,
    off_gas,
    listed_buildings,
)


def parse_args():
    """
    Create ArgumentParser and passes arguments to `main()` and runs `main()`.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--epc_path", help="S3 URI to EPC dataset", type=str, required=True
    )

    parser.add_argument(
        "--save_as",
        help="S3 path to save enhanced EPC dataset to",
        type=str,
        default=None,
        required=False,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    epc_path = args.epc_path
    save_as = args.save_as

    # Import processed EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    epc_df = pl.read_parquet(epc_path)

    # Add feature: lat/long
    logging.info("Adding lat/lon data to EPC")
    uprn_latlon_df = lat_lon.transform_df_osopen_uprn_latlon()
    enhanced_epc_df = epc_df.join(uprn_latlon_df, how="left", on="UPRN")

    # Replace `lad_code` from postcode with `lad_code` from geospatial join and postcode
    logging.info("Adding LAD code with geospatial join")
    uprn_lad_df = output_areas.sjoin_df_uprn_lad_code(enhanced_epc_df)
    enhanced_epc_df = enhanced_epc_df.drop("lad_code").join(
        uprn_lad_df, how="left", on="UPRN"
    )

    # Join new features to EPC dataset
    # Add feature: garden space avg
    logging.info("Adding average garden size per MSOA to EPC")
    garden_space_avg_msoa_df = garden_space_avg.generate_df_garden_space_avg()
    enhanced_epc_df = epc.add_col_msoa_avg_outdoor_space_property_type(enhanced_epc_df)
    enhanced_epc_df = enhanced_epc_df.join(
        garden_space_avg_msoa_df,
        how="left",
        left_on=["msoa", "msoa_avg_outdoor_space_property_type"],
        right_on=["MSOA code", "msoa_avg_outdoor_space_property_type"],
    )

    # Add feature: building conservation area flag
    logging.info("Adding building conservation area flag")
    # Get UPRNs in building conservation areas
    uprns_in_cons_area_df = conservation_areas.generate_df_uprn_to_cons_area(
        enhanced_epc_df
    )
    enhanced_epc_df = enhanced_epc_df.join(uprns_in_cons_area_df, how="left", on="UPRN")

    # Label local authorities with missing building conservation area data
    lad_cons_areas_df = (
        conservation_areas.generate_df_conservation_area_data_availability(
            ladcd_col="LAD23CD"
        )
    )
    enhanced_epc_df = enhanced_epc_df.join(
        lad_cons_areas_df, how="left", left_on="lad_code", right_on="LAD23CD"
    )

    # Add feature: property density
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

    # Add feature: off gas postcodes
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
    if not save_as:
        save_as = f"s3://asf-heat-pump-suitability/outputs/{datetime.today().strftime('%Y%m%d')}_2023_Q2_EPC_weighted_features.parquet"
    fs = s3fs.S3FileSystem()
    with fs.open(save_as, mode="wb") as f:
        enhanced_epc_df.write_parquet(f)
