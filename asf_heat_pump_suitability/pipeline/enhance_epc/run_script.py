import logging
import polars as pl
import s3fs
import pathlib
from tqdm import tqdm
import time
from typing import Optional
from argparse import ArgumentParser
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.enhance_epc import enhance_epc
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target, reweight_epc


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
    Enhance EPC dataset with additional features: LSOA; MSOA.

    Args
        epc_path (str): S3 URI to EPC dataset
        save_output (str): S3 path to save enhanced EPC dataset to. Optional.

    Returns
        pl.DataFrame: enhanced EPC dataset with additional features
    """
    # Import processed EPC
    logging.info(f"Loading EPC file from path: {epc_path}")
    if pathlib.Path(epc_path).suffixes == ".csv":
        fs = s3fs.S3FileSystem()
        with fs.open(epc_path, mode="rb") as f:
            epc_df = pl.read_csv(f, columns=config["usecols"]["epc"])
    else:
        epc_df = pl.read_parquet(epc_path, columns=config["usecols"]["epc"])

    # Join ONSPD LSOA col
    enhanced_epc_df = enhance_epc.join_df_additional_features(epc_df)

    # Reweight EPC
    features = [
        "property_type",
        "build_year",
        "tenure",
    ]  # TODO: add nrooms when categories collapsed
    enhanced_epc_df = epc_df.drop_nulls(subset=["lsoa"])
    enhanced_epc_df = reweight_epc.add_cols_weighting_features(enhanced_epc_df)
    enhanced_epc_df = reweight_epc.drop_nulls_feature_cols(
        df=enhanced_epc_df, features=features
    )

    lsoas = enhanced_epc_df["lsoa"].unique()
    target_marginals = prepare_target.get_dict_target_marginals()
    weights = {"UPRN": [], "weight": []}
    lsoa_stats = {"lsoa": [], "time": [], "lost_rows": []}

    for lsoa in tqdm(lsoas):
        try:
            print(lsoa)
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
            reweighted_sample = reweight_epc.generate_reweighted_sample(
                balance_sample=sample, balance_target=target
            )
            _uprns, _weights = reweight_epc.get_tuple_sample_weights(
                reweighted_sample=reweighted_sample
            )
            weights["UPRN"].extend(_uprns)
            weights["weight"].extend(_weights)
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
