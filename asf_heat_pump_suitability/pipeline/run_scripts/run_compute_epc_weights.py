"""
Weight properties with Iterative Proportional Fitting per LSOA / Data Zone according to the
following features:
- property type (detached, semi-detached, terraced, flats, other);
- tenure (owner-occupied, social rental, private rental)
- build year (pre- and post-1930 split, and unknown); [applies to England and Wales only*]

*Data Zones in Scotland are reweighted on two features only (property type and tenure) due to the absence of target
build year data aggregated to Data Zone-level.

To run:
python -i asf_heat_pump_suitability/pipeline/run_scripts/run_compute_epc_weights.py --epc [path/to/EPC] -y [YYYY] -q [N]

NB: this pipeline takes the preprocessed and deduplicated EPC dataset in parquet file format.
"""

import logging
import polars as pl
import s3fs
from tqdm import tqdm
from datetime import datetime
import time
import argparse
from asf_heat_pump_suitability.pipeline.prepare_features import output_areas
from asf_heat_pump_suitability.pipeline.reweight_epc import (
    prepare_target,
    prepare_sample,
    reweight_epc,
)


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse arguments.

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
    # Set reweighting features for each nation
    COUNTRY_FEATURES = {
        "Scotland": ["property_type", "tenure"],
        "England": ["property_type", "tenure", "build_year"],
        "Wales": ["property_type", "tenure", "build_year"],
    }

    args = parse_arguments()
    epc_path = args.epc
    year = args.year
    q = args.quarter
    save_as = args.save_as

    # Import processed & deduplicated EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    epc_df = pl.read_parquet(
        epc_path,
        columns=[
            "UPRN",
            "POSTCODE",
            "COUNTRY",
            "TENURE",
            "PROPERTY_TYPE",
            "BUILT_FORM",
            "CONSTRUCTION_AGE_BAND",
        ],
    )

    # Join ONSPD LSOA col
    epc_df = output_areas.standardise_col_postcode(epc_df, pcd_col="POSTCODE")
    lsoa_df = output_areas.load_transform_df_lsoas()
    epc_df = epc_df.join(lsoa_df, how="left", on="POSTCODE")

    # Reweight EPC
    # 1. Add standardised weighting feature columns to EPC and drop rows missing data required for reweighting
    epc_df = epc_df.drop_nulls(subset=["lsoa"])
    epc_df = prepare_sample.add_cols_weighting_features(epc_df)

    # 2. Prepare results dicts
    weights = {"UPRN": [], "lsoa": [], "weight": [], "proportional_weight": []}
    lsoa_stats = {"lsoa": [], "time": [], "lost_rows": []}

    for key, features in COUNTRY_FEATURES.items():
        logging.info(
            f"Running reweighting for {key}. Reweighting using the following features: {features}"
        )
        epc_cleaned_df = epc_df.filter(pl.col("COUNTRY") == key)
        epc_cleaned_df = prepare_sample.drop_nulls_feature_cols(
            df=epc_cleaned_df, features=features
        )

        # 3. Generate target marginals for all features and LSOAs
        target_marginals = prepare_target.get_dict_target_marginals(features=features)

        # 4. Reweight properties per LSOA
        for lsoa in tqdm(epc_cleaned_df["lsoa"].unique()):
            try:
                start = time.time()
                sample, lost_rows = reweight_epc.generate_balance_sample(
                    df=epc_cleaned_df,
                    features=features,
                    lsoa=lsoa,
                    target_marginals=target_marginals,
                )
                target = prepare_target.generate_balance_target_population(
                    target_marginals=target_marginals, lsoa=lsoa
                )
                weighted_sample = reweight_epc.generate_weighted_sample(
                    balance_sample=sample, balance_target=target
                )
                _weights = reweight_epc.get_dict_sample_weights(
                    weighted_sample=weighted_sample
                )

                # Add outputs weights for LSOA to dict
                weights["UPRN"].extend(_weights["UPRN"])
                # Adding LSOA required for dummy rows
                weights["lsoa"].extend([lsoa for i in range(len(_weights["UPRN"]))])
                weights["weight"].extend(_weights["weight"])
                weights["proportional_weight"].extend(_weights["proportional_weight"])

                # LSOA stats
                end = time.time()
                lsoa_stats["lsoa"].append(lsoa)
                lsoa_stats["time"].append(end - start)
                lsoa_stats["lost_rows"].append(lost_rows)

            except KeyError:
                logging.warning(f"No target data found for LSOA: {lsoa}. Skipping.")
                continue

    # Get df of UPRNs, reweighting features, and weights for all nations
    weights = pl.DataFrame(weights)
    epc_df = epc_df.select(["UPRN", "property_type", "tenure", "build_year"])
    # Left join ensures we retain dummy rows which we need to retain for reweighting evaluation
    weights = weights.join(epc_df, how="left", on="UPRN")

    lsoa_stats_df = pl.DataFrame(lsoa_stats)

    # 5. Save to S3
    if not save_as:
        save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_weights"
    fs = s3fs.S3FileSystem()

    # Save weighted EPC
    with fs.open(
        f"{save_as}.parquet",
        mode="wb",
    ) as f:
        weights.write_parquet(f)

    # Save weighting stats
    with fs.open(
        f"{save_as}_stats.parquet",
        mode="wb",
    ) as f:
        lsoa_stats_df.write_parquet(f)
