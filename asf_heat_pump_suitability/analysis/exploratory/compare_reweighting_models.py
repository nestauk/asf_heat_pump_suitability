"""
Run reweighting with 2 features (property type, tenure); 3 features (+ build year); and 3 features with multi-level
(LSOA- and LA-level) target data, and save results to S3.

To run:
python -i asf_heat_pump_suitability/pipeline/run_scripts/run_compute_epc_weights.py --epc_path [path/to/unweighted/EPC] -y [YYYY] -q [N]
"""

import logging
import polars as pl
import s3fs
import pathlib
from tqdm import tqdm
from datetime import datetime
import time
import argparse
from asf_heat_pump_suitability import config
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
        "--epc_path",
        help="S3 URI to processed and deduplicated EPC dataset",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    epc_path = args.epc_path
    year = args.year
    q = args.quarter

    # Import processed & deduplicated EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    if ".csv" in pathlib.Path(epc_path).suffixes:
        epc_df = pl.read_csv(epc_path, columns=config["usecols"]["epc"])
    else:
        epc_df = pl.read_parquet(epc_path, columns=config["usecols"]["epc"])

    # Join ONSPD LSOA col
    epc_df = output_areas.standardise_col_postcode(epc_df, pcd_col="POSTCODE")
    onspd_df = output_areas.transform_df_ons_pd()
    epc_df = epc_df.join(onspd_df, how="left", on="POSTCODE")

    # Reweight EPC
    # 1. Add standardised weighting feature columns to EPC and drop rows missing data required for reweighting
    epc_df = epc_df.drop_nulls(subset=["lsoa"])
    epc_df = prepare_sample.add_cols_weighting_features(epc_df)

    feature_composition = {
        "2_features": ["property_type", "tenure"],
        "3_features": ["property_type", "tenure", "build_year"],
        "3_features_mixed_lsoa_la": ["property_type", "tenure", "build_year"],
    }

    for key, features in feature_composition.items():
        logging.info(f"Running reweighting with {key}: {features}")
        epc_cleaned_df = prepare_sample.drop_nulls_feature_cols(
            df=epc_df, features=features
        )

        # 2. Generate target marginals for all features and LSOAs
        if key == "3_features_mixed_lsoa_la":
            target_marginals = prepare_target.get_dict_target_marginals(
                features=features, use_la_build_year=True
            )
        else:
            target_marginals = prepare_target.get_dict_target_marginals(
                features=features, use_la_build_year=False
            )

        # 3. Prepare results dicts
        weights = {"UPRN": [], "weight": [], "proportional_weight": []}
        lsoa_stats = {"lsoa": [], "time": [], "lost_rows": []}

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
                weights["weight"].extend(_weights["weight"])
                weights["proportional_weight"].extend(_weights["proportional_weight"])

                # LSOA stats
                end = time.time()
                lsoa_stats["lsoa"].append(lsoa)
                lsoa_stats["time"].append(end - start)
                lsoa_stats["lost_rows"].append(lost_rows)

            except KeyError:
                logging.info(f"No target data found for LSOA: {lsoa}. Skipping.")
                continue

        weights = pl.DataFrame(weights)
        # Outer join so the dummy rows are still included (these will have a UPRN prefixed with 'dummy_')
        df = epc_cleaned_df.join(weights, how="full", on="UPRN")
        lsoa_stats_df = pl.DataFrame(lsoa_stats)

        # 5. Save to S3
        save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_NE_sample_weighted_{key}"
        fs = s3fs.S3FileSystem()

        # Save weighted EPC
        with fs.open(
            f"{save_as}.parquet",
            mode="wb",
        ) as f:
            df.write_parquet(f)

        # Save weighting stats
        with fs.open(
            f"{save_as}_stats.parquet",
            mode="wb",
        ) as f:
            lsoa_stats_df.write_parquet(f)
