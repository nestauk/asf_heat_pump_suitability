"""
Evaluate the effect of reweighting with three different models by calculating the errors before and after reweighting.
The three models are:
- 2 features for reweighting: property type and tenure
- 3 features: property type; tenure; build year
- 3 mixed-level features: property type and tenure at LSOA level; build year at local authority level.

Errors are calculated between the sample (unweighted and weighted) and target proportions of properties for each feature per LSOA.

Run with:

python -i asf_heat_pump_suitability/analysis/exploratory/run_evaluate_reweighting_models.py -y [YYYY] -q [N]
"""

import polars as pl
from tqdm import tqdm
import argparse

from asf_heat_pump_suitability.pipeline.evaluation import evaluate_reweighting
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target
from asf_heat_pump_suitability.pipeline.prepare_features.epc import clean_df_nrooms
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


def get_errors_for_lsoa(
    feature_name: str,
    lsoa_code: str,
    target_features: dict,
    target_marginals: dict,
    epc_subset: pl.DataFrame,
    error_metric_names: list = [
        "rmse_no_missing_cats",
        "mae_no_missing_cats",
        "rmse_missing_cats",
        "mae_missing_cats",
    ],
) -> dict:
    """
    Get errors between target and unweighted, and target and weighted, for one feature and one LSOA.

    Args:
                    feature_name (str): The feature name to calculate errors for
                    lsoa_code (str): The LSOA
                    target_features (dict): A nested dictionary of the counts for all property features and LSOAs
                        For example, {'tenure': {'E01022833': {'owner-occupied': 548, 'rental (social)': 53}, 'E01013414': {}}, 'property_type': {}}
                    target_marginals (dict): A nested dictionary of the proportions for all property features and LSOAs
                    epc_subset (pl.DataFrame): EPC dataset for this LSOA
                    error_metric_names (list): The error metric names desired in the output, defined in reweighting.get_error_metrics.
                        Defaults to ["rmse_no_missing_cats","mae_no_missing_cats","rmse_missing_cats","mae_missing_cats",]

    Returns:
                    dict: A dictionary containing various metrics for both the unweighted and weighted EPC data compared to
                                    the target dataset for this feature.t

    """

    # Find target values for this feature and LSOA
    target_counts = target_features[feature_name].get(lsoa_code)
    target_proportions = target_marginals[feature_name].get(lsoa_code)

    if target_counts:
        # If we have target information then calculate the errors

        # Calculate the errors between the original EPC data and the target data
        orig_counts_dict = epc_subset[feature_name].value_counts().to_dict()
        orig_counts = dict(
            zip(orig_counts_dict[feature_name], orig_counts_dict["count"])
        )
        orig_proportions = evaluate_reweighting.calculate_proportions(orig_counts)
        error_metrics_orig = evaluate_reweighting.get_error_metrics(
            orig_counts, orig_proportions, target_counts, target_proportions
        )
        if not all(epc_subset["weight"].is_null()):
            # Get the sum of the weights for each feature category (not technically a 'count')
            reweighted_counts_dict = (
                epc_subset.group_by(feature_name).agg(pl.col("weight").sum()).to_dict()
            )
            reweighted_counts = dict(
                zip(
                    reweighted_counts_dict[feature_name],
                    reweighted_counts_dict["weight"],
                )
            )
            reweighted_proportions = evaluate_reweighting.calculate_proportions(
                reweighted_counts
            )
            error_metrics_reweighted = evaluate_reweighting.get_error_metrics(
                reweighted_counts,
                reweighted_proportions,
                target_counts,
                target_proportions,
            )
            # Calculate the average error reduction from the original to reweighted relative to the original proportions
            average_error_reduction = evaluate_reweighting.get_error_reduction(
                orig_proportions, reweighted_proportions, target_proportions
            )

        else:
            logger.info(
                f"All weights are null for the {feature_name} feature in LSOA {lsoa_code}"
            )
            error_metrics_reweighted = {}
            average_error_reduction = None
    else:
        logger.info(
            f"No target data for the {feature_name} feature in LSOA {lsoa_code}"
        )
        return None

    final_metrics = {}
    for metric in error_metric_names:
        final_metrics[metric] = {
            "unweight": error_metrics_orig.get(metric, None),
            "reweight": error_metrics_reweighted.get(metric, None),
            "error_reduction": average_error_reduction,
        }
    return final_metrics


def load_epc_reweights(
    reweighted_path: str = "s3://asf-heat-pump-suitability/outputs/2023_Q2_EPC_enhanced_weights.parquet",
) -> pl.DataFrame:
    """
    Load and slightly clean the reweighted EPC data

    Args:
                    reweighted_path (str): The S3 location of the EPC data enhanced with weights

    Returns:
                    reweighted_epc_df (pl.DataFrame): The EPC data enhanced with weights
    """

    reweighted_epc_df = pl.read_parquet(reweighted_path)
    reweighted_epc_df = clean_df_nrooms(reweighted_epc_df)

    return reweighted_epc_df


def filter_epc(
    reweighted_epc_df: pl.DataFrame,
    target_features: dict,
    evaluation_feature_cols: list,
) -> pl.DataFrame:
    """
    Filter the EPC dataset to just what is needed for calculating errors on.
    This helps with processing time.

    Args:
                    reweighted_epc_df (pl.DataFrame): The EPC data enhanced with weights
                    target_features (dict): A nested dictionary of feature counts for LSOAs in the target dataset
                    evaluation_feature_cols (list): The list of EPC feature columns to evaluate on

    Returns:
                    reweighted_epc_df (pl.DataFrame): The filtered EPC data enhanced with weights
    """

    # Get all the LSOA's in the target data
    all_target_lsoas = set()
    for k, v in target_features.items():
        all_target_lsoas.update(set(v.keys()))
    # No need to try to find errors if a LSOA isn't in this data
    reweighted_epc_df_with_target = reweighted_epc_df.filter(
        pl.col("lsoa").is_in(all_target_lsoas)
    )

    # Remove unneeded columns to speed up processing
    reweighted_epc_df_with_target = reweighted_epc_df_with_target.select(
        pl.col(evaluation_feature_cols + ["lsoa", "weight", "proportional_weight"])
    )

    return reweighted_epc_df_with_target


def load_df_reweighted_epc(feature_composition: str) -> pl.DataFrame:
    """
    Load dataframe of EPC sample data reweighted with specified feature composition. Sample is of local authorities in
    England and Wales north of southern Liverpool.

    Args:
        feature_composition (str): composition of features used for reweighting. Options: `2_features` for property type
        and tenure only; `3_features` for the latter plus build year; `3_features_mixed_lsoa_la` for multi-level 3 features
        (LSOA and LA-level, where build year is LA-level data)

    Returns:
        pl.DataFrame: reweighted EPC data
    """
    path = f"s3://asf-heat-pump-suitability/outputs/2023Q4/20241030_2023_Q4_EPC_NE_sample_weighted_{feature_composition}.parquet"

    return pl.read_parquet(path)


if __name__ == "__main__":

    args = parse_arguments()
    year = args.year
    q = args.quarter

    error_metric_names = [
        "rmse_no_missing_cats",
        "mae_no_missing_cats",
        "rmse_missing_cats",
        "mae_missing_cats",
    ]

    feature_composition = [
        "2_features",
        "3_features",
        "3_features_mixed_lsoa_la",
    ]

    features = ["property_type", "tenure", "build_year"]

    for composition in feature_composition:

        target_features = prepare_target.get_dict_dfs_counts(
            features=features
        )  # Counts
        target_marginals = prepare_target.get_dict_target_marginals(features=features)

        target_features = {
            k: prepare_target.to_dict_feature_marginals(v)
            for k, v in target_features.items()
        }

        reweighted_epc_df = load_df_reweighted_epc(feature_composition=composition)

        reweighted_epc_df_with_target = filter_epc(
            reweighted_epc_df, target_features, features
        )

        logger.info(
            f"Evaluating for {reweighted_epc_df_with_target['lsoa'].n_unique()} LSOAs"
        )

        full_results = {}
        for lsoa_code, epc_subset in tqdm(
            reweighted_epc_df_with_target.group_by("lsoa")
        ):
            lsoa_code = lsoa_code[0]
            feature_results = {}
            for feature_name in features:
                results = get_errors_for_lsoa(
                    feature_name,
                    lsoa_code,
                    target_features,
                    target_marginals,
                    epc_subset,
                    error_metric_names,
                )
                feature_results[feature_name] = results
            feature_results["num_props"] = len(epc_subset)
            feature_results["reweighted"] = not all(epc_subset["weight"].is_null())
            full_results[lsoa_code] = feature_results

        # Save to S3
        name = (
            f"outputs/{year}Q{q}/{year}_Q{q}_EPC_weights_{composition}_evaluation.json"
        )
        save_to_s3("asf-heat-pump-suitability", full_results, name)
