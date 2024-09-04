"""
Temporary script to process garden size data for 2023 Q4 EPC garden size estimates calculated from `run_calculate_garden_size.py`.
"""

import polars as pl
import pandas as pd
import argparse
from datetime import datetime
from asf_heat_pump_suitability.utils import save_utils
import logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epc_path",
        help="Path to parquet with EPC properties with added features and weights to join garden size to.",
        required=True,
    )

    parser.add_argument(
        "--gardens_path",
        help="Path to parquet with estimated garden sizes per UPRN. Defaults to concatenating existing garden size files.",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    if not args.gardens_path:
        # Concatenate garden size estimate files
        path1 = "s3://asf-heat-pump-suitability/outputs/2023Q4/20240901_2023_Q4_EPC_garden_size_estimates_01.parquet"
        path2 = "s3://asf-heat-pump-suitability/outputs/2023Q4/20240902_2023_Q4_EPC_garden_size_estimates_02.parquet"

        logger.info(f"Loading dataframe from: {path1}")
        df1 = pd.read_parquet(path1)
        logger.info(f"Loading dataframe from: {path2}")
        df2 = pd.read_parquet(path2)

        logger.info("Concatenating dataframes")
        gardens_df = pd.concat([df1, df2])

        del df1, df2

        # Drop duplicate rows (i.e. there is some overlap between the two dataframes which we want to remove)
        logger.info("Dropping duplicate rows")
        usecols = [col for col in gardens_df.columns if col != "building_ids"]
        gardens_df = gardens_df.drop_duplicates(usecols)
        logger.info("Saving dataframe to S3")
        gardens_df.to_parquet(
            f"s3://asf-heat-pump-suitability/outputs/2023Q4/{datetime.today().strftime('%Y%m%d')}_2023_Q4_EPC_garden_size_estimates_complete.parquet"
        )
        gardens_df = pl.from_pandas(gardens_df)

    else:
        logger.info(f"Loading dataframe from: {args.garden_path}")
        gardens_df = pl.read_parquet(args.gardens_path)

    # DEDUPLICATE
    # Some UPRNs are matched to multiple gardens (land parcels) due to erroneous land data
    # We will deduplicate them by taking the average size of the multiple gardens (for gardens below a threshold size)
    gardens_df = gardens_df.select(
        [
            "UPRN",
            "NATIONALCADASTRALREFERENCE",
            "garden_area_m2",
        ]
    ).with_columns(pl.col(pl.Float64).round(2))

    # Remove gardens with area above the 97th percentile if they are matched to duplicate UPRNs
    gardens_df = gardens_df.with_columns(
        pl.col("UPRN").is_duplicated().alias("UPRN_duplicated")
    )
    logger.info("Calculating median garden size of UPRNs with multiple garden matches")
    gardens_df = gardens_df.filter(
        ~(
            pl.col("UPRN_duplicated")
            & (
                pl.col("garden_area_m2")
                > gardens_df["garden_area_m2"].quantile(quantile=0.97)
            )
        )
        # (pl.col("garden_area_m2") < gardens_df["garden_area_m2"].quantile(quantile=0.97))  # Alternatively, remove all gardens above 97th percentile
    )
    # Calculate median garden size for UPRNs with multiple gardens
    gardens_df = gardens_df.group_by("UPRN").agg(pl.median("garden_area_m2"))

    # Load EPC data and join to garden size
    logger.info(f"Loading EPC data from: {args.epc_path}")
    epc_df = pl.read_parquet(args.epc_path)
    logger.info("Joining garden size to EPC data")
    epc_df = epc_df.join(gardens_df, how="left", on="UPRN")
    epc_df = epc_df.with_columns(
        pl.col("garden_area_m2")
        .fill_null(pl.col("msoa_avg_outdoor_space_m2"))
        .alias("garden_area_m2")
    )
    logger.info("Saving updated EPC dataset to S3")
    save_as = f"s3://asf-heat-pump-suitability/outputs/2023Q4/{datetime.today().strftime('%Y%m%d')}_2023_Q4_EPC_weighted_features_gardens.parquet"
    save_utils.save_parquet_to_s3(epc_df, save_as)
