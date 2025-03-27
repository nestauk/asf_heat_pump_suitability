import polars as pl
from collections import OrderedDict
import argparse
import logging
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.analysis.flats_on_fossils.features import (
    building_rise,
    fuel_type,
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
        help="Path to save output file with garden size per EPC record to. If unspecified, save with default filename.",
        type=str,
        required=False,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    epc_schema = OrderedDict(
        [
            ("UPRN", pl.String),
            ("ADDRESS1", pl.String),
            ("ADDRESS2", pl.String),
            ("POSTCODE", pl.String),
            ("PROPERTY_TYPE", pl.String),
            ("FLAT_STOREY_COUNT", pl.String),
            ("MAIN_FUEL", pl.String),
            ("MAINHEAT_DESCRIPTION", pl.String),
        ]
    )

    logging.info("Loading EPC data")
    epc_df = pl.read_parquet(
        args.epc,
        columns=[
            "UPRN",
            "ADDRESS1",
            "ADDRESS2",
            "POSTCODE",
            "PROPERTY_TYPE",
            "FLAT_STOREY_COUNT",
            "MAIN_FUEL",
            "MAINHEAT_DESCRIPTION",
        ],
    ).cast(epc_schema)

    logging.info("Loading EPC building footprint data")
    epc_footprint_df = pl.read_parquet(
        f"s3://asf-heat-pump-suitability/outputs/{args.year}Q{args.quarter}/analysis/{args.year}_Q{args.quarter}_epc_building_footprints.parquet"
    )
    epc_footprint_df = building_rise.filter_df_epc_building_footprints(epc_footprint_df)
    epc_footprint_df = building_rise.deduplicate_df_epc_building_footprints(
        epc_footprint_df
    )
    logging.info(
        f"Building height data available for {len(epc_footprint_df)} EPC records"
    )

    logging.info("Filtering EPC data to flats and maisonettes only")
    epc_df = epc_df.filter(pl.col("PROPERTY_TYPE").is_in(["Flat", "Maisonette"]))

    logging.info("Joining building footprint data to EPC")
    epc_df = epc_df.join(epc_footprint_df, how="left", on="UPRN")

    logging.info("Adding fuel type information")
    epc_df = fuel_type.extend_df_fuel_type(epc_df)

    if not args.save_as:
        args.save_as = f"s3://asf-heat-pump-suitability/outputs/{args.year}Q{args.quarter}/analysis/{args.year}_Q{args.quarter}_epc_flats_processed.parquet"

    save_utils.save_to_s3(epc_df, args.save_as)
