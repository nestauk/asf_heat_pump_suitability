"""
Add new features to EPC dataset:
- mean average garden size per MSOA
- lat/lon per UPRN
- property density per LSOA
- off gas properties by postcode
- listed building status per UPRN
- England and Wales building conservation area flag per UPRN
- Scotland World Heritage Site flag per UPRN
- grid capacity per LSOA (% of homes which could install a HP with current grid capacity)
- presence of anchor properties per LSOA

To run:
python -i asf_heat_pump_suitability/pipeline/run_scripts/run_add_features.py --epc [path/to/EPC] -y [YYYY] -q [Q]

NB: this pipeline takes the preprocessed and deduplicated EPC dataset in parquet file format.
"""

import logging
import polars as pl
import argparse
from datetime import datetime
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_sample
from asf_heat_pump_suitability.pipeline.prepare_features import (
    anchor_properties,
    protected_areas,
    epc,
    garden_space_avg,
    lat_lon,
    output_areas,
    property_density,
    off_gas,
    listed_buildings,
    grid_capacity,
)


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epc",
        help="Path to processed and deduplicated EPC dataset in parquet file format",
        type=str,
        required=True,
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
    epc_path = args.epc
    save_as = args.save_as
    year = args.year
    q = args.quarter

    # Import processed EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    epc_df = pl.read_parquet(
        epc_path,
        columns=[
            "UPRN",
            "COUNTRY",
            "POSTCODE",
            "PROPERTY_TYPE",
            "BUILT_FORM",
            "CURRENT_ENERGY_RATING",
        ],
    )

    logging.info("Add LSOA, MSOA, LAD, and rural-urban indicators to EPC")
    epc_df = output_areas.standardise_col_postcode(epc_df, pcd_col="POSTCODE")
    onspd_df = output_areas.load_transform_df_area_info()
    epc_df = epc_df.join(onspd_df, how="left", on="POSTCODE")

    logging.info("Adding lat/lon data to EPC")
    uprn_latlon_df = lat_lon.transform_df_osopen_uprn_latlon()
    epc_df = epc_df.join(uprn_latlon_df, how="left", on="UPRN")
    epc_gdf = lat_lon.generate_gdf_uprn_coords(
        epc_df, usecols=["UPRN", "COUNTRY", "lad_code"]
    )

    # TODO this is far too slow and is only used to correct some EPC records with incorrect postcodes which get joined to the wrong LSOA/MSOA
    # TODO can we filter the dataset somehow and only do the join with incorrect postcodes?
    # # Replace `lad_code` from postcode with `lad_code` from geospatial join and postcode
    # logging.info("Adding LAD code with geospatial join")
    # uprn_lad_df = output_areas.sjoin_df_uprn_lad_code(epc_gdf)
    # epc_df = epc_df.drop("lad_code").join(uprn_lad_df, how="left", on="UPRN")

    logging.info("Adding property type to EPC")
    # Below we process the EPC `PROPERTY_TYPE` column and rename, then drop the original column
    epc_df = prepare_sample.add_col_property_type(epc_df).drop(
        ["PROPERTY_TYPE", "BUILT_FORM"]
    )

    logging.info("Adding average garden size per MSOA to EPC")
    garden_space_avg_msoa_df = garden_space_avg.generate_df_garden_space_avg()
    epc_df = epc.add_col_msoa_avg_outdoor_space_property_type(epc_df)
    epc_df = epc_df.join(
        garden_space_avg_msoa_df,
        how="left",
        left_on=["msoa", "msoa_avg_outdoor_space_property_type"],
        right_on=["MSOA code", "msoa_avg_outdoor_space_property_type"],
    )

    logging.info("Adding protected area flag")
    uprns_in_protected_area_df = (
        protected_areas.load_transform_df_uprn_in_protected_area(epc_gdf)
    )
    epc_df = epc_df.join(
        uprns_in_protected_area_df, how="left", on="UPRN"
    ).with_columns(pl.col("in_protected_area").fill_null(False))

    logging.info(
        "Adding local authority building conservation area data availability flag for England and Wales"
    )
    lad_cons_areas_df = protected_areas.generate_df_conservation_area_data_availability(
        ladcd_col="LAD23CD"
    )
    epc_df = epc_df.join(
        lad_cons_areas_df, how="left", left_on="lad_code", right_on="LAD23CD"
    )

    logging.info("Adding property density to EPC")
    lsoa_density_df = property_density.generate_df_property_density()
    epc_df = epc_df.join(lsoa_density_df, how="left", on="lsoa")

    logging.info("Adding off gas grid column to EPC")
    off_gas_postcodes = off_gas.process_off_gas_data()
    epc_df = off_gas.add_off_gas_feature(epc_df, off_gas_postcodes)

    logging.info("Adding listed buildings to EPC")
    listed_buildings_df = listed_buildings.generate_df_epc_listed_buildings(
        epc_df=epc_df
    )
    epc_df = epc_df.join(listed_buildings_df, how="left", on="UPRN").with_columns(
        pl.col("listed_building").fill_null(False)
    )

    logging.info("Adding grid capacity column to EPC")
    grid_capacities = grid_capacity.calculate_grid_capacity().select(
        ["lsoa", "heatpump_installation_percentage"]
    )
    epc_df = epc_df.join(grid_capacities, how="left", on="lsoa")

    logging.info("Adding anchor properties column to EPC")
    anchor_properties_df = anchor_properties.identify_anchor_properties_df().select(
        ["lsoa", "has_anchor_property"]
    )
    epc_df = epc_df.join(anchor_properties_df, how="left", on="lsoa").with_columns(
        pl.col("has_anchor_property").fill_null(False)
    )

    # Save to S3
    if not save_as:
        save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_features.parquet"
    save_utils.save_to_s3(epc_df, save_as)
