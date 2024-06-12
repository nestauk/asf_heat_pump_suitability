"""
Evaluate the effect of reweighting by calculating the errors before and after reweighting.

Errors are calculated between the sample (unweighted and weighted) and target proportions of properties for each feature per LSOA.

Run with:

python asf_heat_pump_suitability/pipeline/reweight_epc/evaluate_reweighting.py
	--reweighted_dir "s3://asf-heat-pump-suitability/outputs/2023_Q2_EPC_enhanced_weights.parquet"
	--sample

[remove the --sample argument to run on full dataset]

"""

import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from argparse import ArgumentParser

from asf_heat_pump_suitability.pipeline.error_analysis.error_analysis_utils import *
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target
from asf_heat_pump_suitability.getters.s3_getters import save_to_s3
from asf_heat_pump_suitability import logger

main_error_metrics = [
    "rmse_all_cats",
    "mae_all_cats",
    "rmse_missing_cats",
    "mae_missing_cats",
]

evaluation_feature_names = ["tenure", "property_type", "build_year", "nrooms"]


def parse_arguments():

    parser = ArgumentParser()

    parser.add_argument(
        "--reweighted_dir",
        help="S3 URI to weighted EPC dataset",
        type=str,
        default="s3://asf-heat-pump-suitability/outputs/2023_Q2_EPC_enhanced_weights.parquet",
    )

    parser.add_argument(
        "--save_output",
        help="S3 path to save evaluation results to",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--sample",
        help="Whether to test this script on a sample or not",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    return args


def get_errors_for_lsoa(
    feature_name: str,
    lsoa_code: str,
    target_features: dict,
    target_marginals: dict,
    epc_subset: pl.DataFrame,
) -> dict:
    """
    Get errors between target and unweighted, and target and weighted, for one feature and one LSOA.

    Args:
                    feature_name (str): The feature name to calculate errors for
                    lsoa_code (str): The LSOA
                    target_features (dict): A nested dictionary of the counts for all property features and LSOAs
                    target_marginals (dict): A nested dictionary of the proportions for all property features and LSOAs
                    epc_subset (pl.DataFrame): EPC dataset for this LSOA

    Returns:
                    dict: A dictionary containing various metrics for both the unweighted and weighted EPC data compared to
                                    the target dataset for this feature.

    """

    target_counts = target_features[feature_name].get(lsoa_code)
    target_proportions = target_marginals[feature_name].get(lsoa_code)

    if target_counts:
        orig_counts_list = epc_subset[feature_name].value_counts().to_dict()
        orig_counts = dict(
            zip(orig_counts_list[feature_name], orig_counts_list["count"])
        )
        orig_proportions = calculate_proportions(orig_counts)
        error_metrics_orig = get_error_metrics(
            orig_counts, orig_proportions, target_counts, target_proportions
        )
        if not all(epc_subset["weight"].is_null()):
            # Get the sum of the weights for each feature category (not technically a 'count')
            reweighted_counts_list = (
                epc_subset.group_by(feature_name).agg(pl.col("weight").sum()).to_dict()
            )
            reweighted_counts = dict(
                zip(
                    reweighted_counts_list[feature_name],
                    reweighted_counts_list["weight"],
                )
            )
            reweighted_proportions = calculate_proportions(reweighted_counts)
            error_metrics_reweighted = get_error_metrics(
                reweighted_counts,
                reweighted_proportions,
                target_counts,
                target_proportions,
            )

            average_error_reduction = get_error_reduction(
                orig_proportions, reweighted_proportions, target_proportions
            )

            prop_reweighted_counts_list = (
                epc_subset.group_by(feature_name)
                .agg(pl.col("proportional_weight").sum())
                .to_dict()
            )
            prop_reweighted_counts = dict(
                zip(
                    prop_reweighted_counts_list[feature_name],
                    prop_reweighted_counts_list["proportional_weight"],
                )
            )
            prop_reweighted_proportions = calculate_proportions(prop_reweighted_counts)
            error_metrics_prop_reweighted = get_error_metrics(
                prop_reweighted_counts,
                prop_reweighted_proportions,
                target_counts,
                target_proportions,
            )

        else:
            # The weights may have been all Null
            error_metrics_reweighted = {}
            error_metrics_prop_reweighted = {}
            average_error_reduction = None
    else:
        error_metrics_orig = {}
        error_metrics_reweighted = {}
        error_metrics_prop_reweighted = {}
        average_error_reduction = None

    final_metrics = {}
    for metric in main_error_metrics:
        final_metrics[metric] = {
            "unweight": error_metrics_orig.get(metric, None),
            "reweight": error_metrics_reweighted.get(metric, None),
            "reweight-prop": error_metrics_prop_reweighted.get(metric, None),
            "error_reduction": average_error_reduction,
        }
    return final_metrics


def load_epc_reweights(
    reweighted_dir: str = "s3://asf-heat-pump-suitability/outputs/2023_Q2_EPC_enhanced_weights.parquet",
) -> pl.DataFrame:
    """
    Load and slightly clean the reweighted EPC data

    Args:
                    reweighted_dir (str): The S3 location of the EPC data enhanced with weights

    Returns:
                    reweighted_epc_df (pl.DataFrame): The EPC data enhanced with weights
    """

    reweighted_epc_df = pl.read_parquet(reweighted_dir)

    # Bit of cleaning to allign with target data
    reweighted_epc_df = reweighted_epc_df.rename({"NUMBER_HABITABLE_ROOMS": "nrooms"})
    reweighted_epc_df = reweighted_epc_df.with_columns(
        pl.col("nrooms").map_elements(
            lambda x: "9+" if x >= 9 else str(x), return_dtype=pl.String
        )
    )
    reweighted_epc_df = reweighted_epc_df.with_columns(
        pl.Series(
            name="nrooms", values=reweighted_epc_df["nrooms"].replace(None, "unknown")
        )
    )
    return reweighted_epc_df


def filter_epc(reweighted_epc_df: pl.DataFrame, target_features: dict) -> pl.DataFrame:
    """
    Filter the EPC dataset to just what is needed for calculating errors on.
    This helps with processing time.

    Args:
                    reweighted_epc_df (pl.DataFrame): The EPC data enhanced with weights
                    target_features (dict): A nested dictionary of feature counts for LSOAs in the target dataset

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
        pl.col(evaluation_feature_names + ["lsoa", "weight", "proportional_weight"])
    )

    return reweighted_epc_df_with_target


if __name__ == "__main__":

    args = parse_arguments()

    target_features = prepare_target.get_dict_dfs_counts()  # Counts
    target_features = {
        k: prepare_target.to_dict_feature_marginals(v)
        for k, v in target_features.items()
    }

    target_marginals = prepare_target.get_dict_target_marginals()

    reweighted_epc_df = load_epc_reweights(args.reweighted_dir)

    reweighted_epc_df_with_target = filter_epc(reweighted_epc_df, target_features)

    if args.sample:
        lsoas = reweighted_epc_df_with_target["lsoa"].unique()
        lsoas = lsoas[0:500]
        reweighted_epc_df_with_target = reweighted_epc_df_with_target.filter(
            pl.col("lsoa").is_in(lsoas)
        )

    logger.info(
        f"Evaluating for {reweighted_epc_df_with_target['lsoa'].n_unique()} LSOAs"
    )

    full_results = {}
    for lsoa_code, epc_subset in tqdm(reweighted_epc_df_with_target.group_by("lsoa")):
        feature_results = {}
        for feature_name in evaluation_feature_names:
            results = get_errors_for_lsoa(
                feature_name, lsoa_code, target_features, target_marginals, epc_subset
            )
            feature_results[feature_name] = results
        feature_results["num_props"] = len(epc_subset)
        feature_results["reweighted"] = not all(epc_subset["weight"].is_null())
        full_results[lsoa_code] = feature_results

        # Save to S3

    if not args.save_output:
        name = args.reweighted_dir.replace("s3://asf-heat-pump-suitability/", "")
        name = f"{name.split('.parquet')[0]}_evaluation.json"
        if args.sample:
            name = f"{name.split('.json')[0]}_sample.json"
        args.save_output = name

    save_to_s3("asf-heat-pump-suitability", full_results, args.save_output)
