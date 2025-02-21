"""
Evaluate the results of reweighting EPC properties against the target data. For English and Welsh LSOAs, we weight the
EPC data on three features: property type; tenure; build year. For Scottish DataZones, we weight the EPC data on only
two (due to no availability of open-source DataZone-level build year data): property type and tenure.

Errors are calculated between the sample and target proportions of properties for each feature per LSOA / DataZone.

Run with:

python -i asf_heat_pump_suitability/pipeline/run_scripts/run_evaluate_reweighting.py -r [path/to/weighted/EPC] -y [YYYY] -q [Q]

"""

import polars as pl
from tqdm import tqdm
import argparse

from asf_heat_pump_suitability.pipeline.evaluation import evaluate_reweighting
from asf_heat_pump_suitability.pipeline.prepare_features import epc
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target
from asf_heat_pump_suitability.getters.s3_getters import save_to_s3
from asf_heat_pump_suitability import logger


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse arguments.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--reweighted_epc",
        help="Path to EPC data with weights in parquet format.",
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
    year = args.year
    q = args.quarter
    full_results_ew = {}
    full_results_s = {}

    error_metrics = [
        "rmse_no_missing_cats",
        "mae_no_missing_cats",
        "rmse_missing_cats",
        "mae_missing_cats",
    ]

    # Set reweighting features for each nation
    country_features = {
        "Scotland": ["property_type", "tenure"],
        "England": ["property_type", "tenure", "build_year"],
        "Wales": ["property_type", "tenure", "build_year"],
    }

    logger.info("Loading reweighted EPC dataset and preparing for evaluation")
    reweighted_df = pl.read_parquet(args.reweighted_epc)
    reweighted_df = epc.extend_df_country_col(df=reweighted_df)

    for country, features in country_features.items():
        _reweighted_df = reweighted_df.select(
            pl.col(features + ["lsoa", "weight", "proportional_weight", "country"])
        ).filter(pl.col("country") == country)

        logger.info(f"Preparing {country} target data for evaluation")
        target_features = prepare_target.get_dict_dfs_counts(features=features)
        target_features = {
            k: prepare_target.to_dict_feature_marginals(v)
            for k, v in target_features.items()
        }
        target_marginals = prepare_target.get_dict_target_marginals(features=features)

        logger.info(
            f"Evaluating {_reweighted_df['lsoa'].n_unique()} LSOAs in {country}"
        )
        for lsoa, subset in tqdm(_reweighted_df.group_by("lsoa")):
            lsoa = lsoa[0]
            feature_results = {}
            for feature_name in features:
                results = evaluate_reweighting.calculate_dict_errors_for_lsoa(
                    feature_name=feature_name,
                    lsoa_code=lsoa,
                    target_features=target_features,
                    target_marginals=target_marginals,
                    epc_subset=subset,
                    error_metric_names=error_metrics,
                )
                feature_results[feature_name] = results
            feature_results["n_properties"] = len(subset)
            if country == "Scotland":
                full_results_s[lsoa] = feature_results
            else:
                full_results_ew[lsoa] = feature_results

    # Save to S3
    for country, results in {"S": full_results_s, "EW": full_results_ew}.items():
        save_as = f"evaluation/reweighting/{year}Q{q}/{year}_Q{q}_EPC_weights_evaluation_{country}.json"
        save_to_s3("asf-heat-pump-suitability", results, save_as)
