"""
Enhance EPC dataset with additional features:
- mean average garden size per MSOA
- lat/lon per UPRN
- property density per LSOA
- off gas properties by postcode
- listed building status per UPRN
- England and Wales building conservation area flag per UPRN

To run:
python asf_heat_pump_suitability/pipeline/run_scripts/run_add_features.py --epc_path [path/to/weighted/EPC] -y [YYYY] -q [N]
"""

import logging
import polars as pl
import s3fs
import argparse
from datetime import datetime
from asf_heat_pump_suitability.pipeline.prepare_features import (
    protected_areas,
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


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epc_path", help="S3 URI to EPC dataset with weights", type=str, required=True
    )

    parser.add_argument(
        "-y",
        "--year",
        help="EPC data year. Format YYYY",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-q",
        "--quarter",
        help="EPC data quarter",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--save_as",
        help="S3 path to save enhanced EPC dataset to. If unspecified, save with default filename.",
        type=str,
        default=None,
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    epc_path = args.epc_path
    save_as = args.save_as
    year = args.year
    q = args.quarter

    # Import processed EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    epc_df = pl.read_parquet(epc_path)

    # Add feature: lat/long
    logging.info("Adding lat/lon data to EPC")
    uprn_latlon_df = lat_lon.transform_df_osopen_uprn_latlon()
    epc_df = epc_df.join(uprn_latlon_df, how="left", on="UPRN")
    epc_gdf = lat_lon.generate_gdf_uprn_coords(epc_df, usecols=["UPRN", "lad_code"])

    # Replace `lad_code` from postcode with `lad_code` from geospatial join and postcode
    logging.info("Adding LAD code with geospatial join")
    uprn_lad_df = output_areas.sjoin_df_uprn_lad_code(epc_gdf)
    epc_df = epc_df.drop("lad_code").join(uprn_lad_df, how="left", on="UPRN")

    # Join new features to EPC dataset
    # Add feature: garden space avg
    logging.info("Adding average garden size per MSOA to EPC")
    garden_space_avg_msoa_df = garden_space_avg.generate_df_garden_space_avg()
    epc_df = epc.add_col_msoa_avg_outdoor_space_property_type(epc_df)
    epc_df = epc_df.join(
        garden_space_avg_msoa_df,
        how="left",
        left_on=["msoa", "msoa_avg_outdoor_space_property_type"],
        right_on=["MSOA code", "msoa_avg_outdoor_space_property_type"],
    )

    # Add feature: building conservation area flag
    logging.info("Adding building conservation area England and Wales flag")
    # Get UPRNs in building conservation areas
    uprns_in_cons_area_df = protected_areas.generate_df_uprn_in_cons_area(epc_gdf)
    epc_df = epc_df.join(uprns_in_cons_area_df, how="left", on="UPRN")

    # Label local authorities with missing building conservation area data
    lad_cons_areas_df = protected_areas.generate_df_conservation_area_data_availability(
        ladcd_col="LAD23CD"
    )
    epc_df = epc_df.join(
        lad_cons_areas_df, how="left", left_on="lad_code", right_on="LAD23CD"
    )

    # Add feature: World Heritage Site flag
    logging.info("Adding World Heritage Site Scotland flag")
    uprns_in_whs_df = protected_areas.generate_df_uprn_in_whs(epc_gdf)
    epc_df = epc_df.join(uprns_in_whs_df, how="left", on="UPRN")

    # Add feature: property density
    logging.info("Adding number of households data to EPC")
    lsoa_number_of_households_df = (
        number_of_households.prepare_df_num_of_households_ons()
    )
    epc_lsoa_number_of_households_df = lsoa_number_of_households_df.select(
        ["lsoa21", "Number of households 2021"]
    )
    epc_df = epc_df.join(epc_lsoa_number_of_households_df, how="left", on="lsoa21")

    logging.info("Adding land area to EPC")
    lsoa_land_area_df = land_area.prepare_df_land_area_ons()
    epc_lsoa_land_area_df = lsoa_land_area_df.select(
        ["lsoa21", "Land Count (Area in KM2)"]
    )
    epc_df = epc_df.join(epc_lsoa_land_area_df, how="left", on="lsoa21")

    logging.info("Adding property density to EPC")
    epc_df = property_density.extend_df_with_property_density(epc_df)

    # Add feature: property density Scotland
    dz_density_df = property_density.generate_df_property_density_s()
    epc_df = epc_df.join(dz_density_df, how="left", right_on="lsoa", left_on="DataZone")

    # Add feature: off gas postcodes
    logging.info("Adding off gas grid column to EPC")
    off_gas_postcodes = off_gas.process_off_gas_data()
    epc_df = off_gas.add_off_gas_feature(epc_df, off_gas_postcodes)

    # Add feature: listed buildings data
    logging.info("Adding listed buildings to EPC")
    listed_buildings_df = listed_buildings.generate_df_epc_listed_buildings(
        epc_df=epc_df
    )
    epc_df = epc_df.join(listed_buildings_df, how="left", on="UPRN")

    # Save to S3
    if not save_as:
        save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_weighted_features.parquet"
    fs = s3fs.S3FileSystem()
    with fs.open(save_as, mode="wb") as f:
        epc_df.write_parquet(f)
