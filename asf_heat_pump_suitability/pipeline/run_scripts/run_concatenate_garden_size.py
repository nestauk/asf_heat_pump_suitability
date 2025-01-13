import polars as pl
import argparse
from datetime import datetime
import logging
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.pipeline.prepare_features import garden_size


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--interim_dir",
        help="Path to S3 directory where interim result for garden size estimates are stored",
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
        "-n",
        "--nations",
        help="Nations that gardens are calculated for. Of England and Wales (ew); Scotland (s); or all (ews).",
        type=str,
        choices=["ew", "s", "ews"],
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
    year = args.year
    q = args.quarter

    interim_files = base_getters.list_obj_s3_location(args.interim_dir)

    # Get df of all EPC records with garden size estimates
    epc_gardens_df = pl.DataFrame()
    for file in interim_files:
        logging.info(f"Loading file: {file}")
        # TODO - delete the string casting when pipeline next rerun as it will be redundant
        df = pl.read_parquet(f"s3://{file}").with_columns(
            pl.col("NATIONALCADASTRALREFERENCE").cast(pl.String)
        )
        # Deduplicate any rows with duplicate data across all columns
        epc_gardens_df = pl.concat([epc_gardens_df, df]).unique()

    if not args.save_as:
        args.save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/gardens/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_garden_size_estimates_{args.nations.upper()}.parquet"
    save_utils.save_to_s3(epc_gardens_df, args.save_as)

    logging.info("Deduplicating UPRNs that were matched to multiple gardens")
    epc_gardens_df = epc_gardens_df.with_columns(pl.col(pl.Float64).round(2))
    epc_gardens_df = garden_size.deduplicate_df_garden_size(epc_gardens_df)
    args.save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/gardens/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_garden_size_estimates_{args.nations.upper()}_deduplicated.parquet"
    save_utils.save_to_s3(epc_gardens_df, args.save_as)
