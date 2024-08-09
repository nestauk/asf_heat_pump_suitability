import logging
import polars as pl
import s3fs
import pathlib
from tqdm import tqdm
import time
from argparse import ArgumentParser
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.prepare_features import output_areas
from asf_heat_pump_suitability.pipeline.reweight_epc import (
    prepare_target,
    prepare_sample,
    reweight_epc,
)


def run():
    """
    Create ArgumentParser and passes arguments to `main()` and runs `main()`.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--epc_path", help="S3 URI to EPC dataset", type=str, required=True
    )

    args = parser.parse_args()

    main(**vars(args))


def main(epc_path: str) -> pl.DataFrame:
    """
    Add LSOA; MSOA data to EPC dataset and weight properties per LSOA according to the following features: property
    type; build year (pre- and post-1930 split); and tenure.

    Args
        epc_path (str): S3 URI to EPC dataset

    Returns
        pl.DataFrame: enhanced EPC dataset with LSOA and MSOA data and weights
    """
    # Import processed & deduplicated EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    if pathlib.Path(epc_path).suffixes == ".csv":
        fs = s3fs.S3FileSystem()
        with fs.open(epc_path, mode="rb") as f:
            epc_df = pl.read_csv(f, columns=config["usecols"]["epc"])
    else:
        epc_df = pl.read_parquet(epc_path, columns=config["usecols"]["epc"])

    # Join ONSPD LSOA col
    epc_df = output_areas.standardise_col_postcode(epc_df, pcd_col="POSTCODE")
    onspd_df = output_areas.transform_df_ons_pd()
    enhanced_epc_df = epc_df.join(onspd_df, how="left", on="POSTCODE")

    # Reweight EPC

    features = [
        "property_type",
        "build_year",
        "tenure",
    ]  # TODO: add nrooms when categories collapsed

    # Add standardised weighting feature columns to EPC and drop rows missing data required for reweighting
    enhanced_epc_df = enhanced_epc_df.drop_nulls(subset=["lsoa"])
    enhanced_epc_df = prepare_sample.add_cols_weighting_features(enhanced_epc_df)
    enhanced_epc_df = prepare_sample.drop_nulls_feature_cols(
        df=enhanced_epc_df, features=features
    )

    lsoas = enhanced_epc_df["lsoa"].unique()
    target_marginals = prepare_target.get_dict_target_marginals()

    # Prepare results dicts
    weights = {"UPRN": [], "weight": [], "proportional_weight": []}
    lsoa_stats = {"lsoa": [], "time": [], "lost_rows": []}

    # Reweight LSOAs
    for lsoa in tqdm(lsoas):
        try:
            start = time.time()
            sample, lost_rows = reweight_epc.generate_balance_sample(
                df=enhanced_epc_df,
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
    enhanced_epc_df = enhanced_epc_df.join(weights, how="left", on="UPRN")
    lsoa_stats_df = pl.DataFrame(lsoa_stats)

    # Save to S3
    fs = s3fs.S3FileSystem()
    with fs.open(
        "s3://asf-heat-pump-suitability/outputs/2023_Q2_EPC_enhanced_weights.parquet",
        mode="wb",
    ) as f:
        enhanced_epc_df.write_parquet(f)

    with fs.open(
        "s3://asf-heat-pump-suitability/outputs/2023_Q2_EPC_enhanced_weights_stats.parquet",
        mode="wb",
    ) as f:
        lsoa_stats_df.write_parquet(f)

    return enhanced_epc_df


if __name__ == "__main__":
    run()
