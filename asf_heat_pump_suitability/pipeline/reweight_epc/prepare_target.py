"""
Functions to prepare target marginals for reweighting with IPF.
"""

import random
import balance
from balance.weighting_methods import rake
import polars as pl
import polars.selectors as cs
from typing import Dict
from asf_heat_pump_suitability.getters import get_target


def get_dict_target_marginals(
    features: list, use_la_build_year: bool = False
) -> Dict[str, Dict[str, dict]]:
    """
    Get nested dictionary containing target proportions of each feature category per LSOA. Primary dictionary keys are
    feature names, secondary keys are LSOA codes, tertiary keys are feature categories.

    Returned nested dict structure:
    ```
        {"tenure": {"lsoa_01": {"owner-occupied": <proportion>, "rental (social)": <proportion>, etc.},
                    "lsoa_02": {"owner-occupied": <proportion>, "rental (social)": <proportion>, etc.}},
         "build_year": {"lsoa_01": {"pre-1930": <proportion>, "post-1930": <proportion>}}
        }
    ```

    Args:
        features (list): features to use in reweighting. Options: tenure; property_type; build_year; n_rooms
        use_la_build_year (bool): use build year data from local authority rather than LSOA

    Returns:
        Dict[str, Dict[str, dict]]: nested dict containing target proportions for each feature category per LSOA
    """
    target_features = get_dict_dfs_counts(
        features=features, use_la_build_year=use_la_build_year
    )

    target_proportions = {
        k: convert_df_proportions(v) for k, v in target_features.items()
    }
    marginals = {k: to_dict_feature_marginals(v) for k, v in target_proportions.items()}

    return marginals


def get_dict_dfs_counts(
    features: list, use_la_build_year: bool = False
) -> Dict[str, pl.DataFrame]:
    """
    Generate dict where keys are feature names and values are dataframes containing counts of each feature variable
    per LSOA in the target datasets.

    Args:
        features (list): features to use in reweighting. Options: tenure; property_type; build_year; n_rooms
        use_la_build_year (bool): use build year data from local authority rather than LSOA

    Returns:
        Dict[str, pl.DataFrame]: dict of dataframes containing counts of each feature variable per LSOA in the target
        datasets
    """
    count_dict = {}

    if "property_type" in features:
        count_dict["property_type"] = (
            get_target.get_df_target_property_type_uncensored()
        )
    if "tenure" in features:
        count_dict["tenure"] = get_target.get_df_target_tenure_uncensored()
    if "build_year" in features:
        if not use_la_build_year:
            count_dict["build_year"] = get_target.get_df_target_build_year()
        else:
            count_dict["build_year"] = get_target.get_df_target_build_year_la()
    if "n_rooms" in features:  # TODO: collapse nrooms categories to increase speed
        count_dict["n_rooms"]: get_target.get_df_target_nrooms()

    return count_dict


def to_dict_feature_marginals(df: pl.DataFrame) -> Dict[str, dict]:
    """
    Convert dataframe with target marginals per LSOA to dict where keys are LSOA codes and values are dictionaries with
    target proportions for each feature variable.

    Returned dict structure:
    ```
        {"lsoa_01": {"owner-occupied": <proportion>, "rental (social)": <proportion>, etc.},
         "lsoa_02": {"owner-occupied": <proportion>, "rental (social)": <proportion>, etc.},
        }
    ```

    Args:
        df (pl.DataFrame): single-feature dataframe with target marginals for each feature variable per LSOA

    Returns:
        Dict[str, dict]: dict with target proportions for each feature variable where keys are LSOA codes
    """
    lsoas = df["lsoa"].to_list()
    marginals = df.select(pl.exclude("lsoa")).to_dicts()
    marginals = dict(zip(lsoas, marginals))

    return marginals


def generate_balance_target_population(
    target_marginals: Dict[str, Dict[str, dict]], lsoa: str
) -> balance.sample_class.Sample:
    """
    Generate artificial target population from target proportions for each feature variable for the specified LSOA.

    Args:
        target_marginals (Dict[str, Dict[str, dict]]): nested dict containing target proportions for each feature
        variable by LSOA. Structure: {"feature_01": {"lsoa_01": {"feature_01_a": <proportion>, "feature_01_b": <proportion>}}}
        lsoa (str): LSOA code to generate target population for

    Returns:
        balance.sample_class.Sample: artificial target population object generated from given marginals for single LSOA
    """
    target = get_dict_lsoa_marginals(target_marginals=target_marginals, lsoa=lsoa)
    target = _get_dict_proportions_sum_one(lsoa_marginals=target, round_n=3)
    df = rake.prepare_marginal_dist_for_raking(target)
    df["weight"] = 1
    df = balance.Sample.from_frame(df, id_column="id", weight_column="weight")
    return df


def get_dict_lsoa_marginals(
    target_marginals: Dict[str, Dict[str, dict]], lsoa: str
) -> Dict[str, dict]:
    """
    Get nested dictionary where keys are feature names and values are dictionaries containing target proportions of each
    feature variable for a single specified LSOA.

    Returned dict structure:
    ```
        {"tenure": {"owner-occupied": <proportion>, "rental (social)": <proportion>, etc.},
         "build_year": {"pre-1930": <proportion>, "post-1930": <proportion>},
         "property_type": {"detached": <proportion>, "semi-detached": <proportion>, etc.}}
        }
    ```

    Args:
        target_marginals (Dict[str, Dict[str, dict]]): nested dict containing target proportions for each feature
        variable by LSOA. Structure: {"feature_01": {"lsoa_01": {"feature_01_a": <proportion>, "feature_01_b": <proportion>}}}
        lsoa (str): LSOA code to get target marginals for

    Returns:
        Dict[str, dict]: nested dictionary containing target proportions for each feature variable for a single specified
        LSOA
    """
    lsoa_marginals = {k: v[lsoa] for k, v in target_marginals.items()}
    return lsoa_marginals


def convert_df_proportions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert dataframe of counts to target proportions for each feature category per LSOA.

    Args:
        pl.DataFrame: dataframe containing counts of each feature category per LSOA

    Returns:
        pl.DataFrame: dataframe of target proportions for each feature category per LSOA
    """
    cols = df.select(cs.integer()).columns
    df = df.with_columns(pl.sum_horizontal(df.select(cols)).alias("total"))

    # Rounding the proportions cuts down the run time to generate target population
    df = df.with_columns(
        [(pl.col(col) / pl.col("total")).round(3).alias(col) for col in cols]
    ).drop("total")

    return df


def _get_dict_proportions_sum_one(
    lsoa_marginals: Dict[str, dict], round_n: int = 3
) -> Dict[str, dict]:
    """
    Ensure target proportions sum to 1 for each feature.

    Args:
        lsoa_marginals (Dict[str, dict]): dict of dicts containing target proportions for each feature category for
        single specified LSOA
        round_n (int): decimal places to round remainder difference to

    Returns:
        Dict[str, dict]: dict of dicts containing target proportions for each feature category for single specified
        LSOA, where proportions add to 1.
    """
    # Make sure rounded proportions add to 1 for each feature
    for feature_name, feature in lsoa_marginals.items():
        # Get remainder from rounding & round remainder, otherwise this can be e.g. 0.0010000001
        diff = round(1 - sum(feature.values()), round_n)

        if diff != 0:

            # If `diff` is negative, i.e. rounded sum of values is >1, then adding `diff` to a very small proportion
            # can create negative proportions. So here we ensure `diff` can only be added to proportions if they are
            # larger than -diff. This prevents any proportions being converted to 0 or negative.
            # If `diff` is positive, it can be added to any category and all categories will be eligible as they are all
            # > -diff.
            # We also ensure that any very small categories will not be significantly changed by the addition of diff.
            # To be eligible for adding diff to, `diff` must be <= 10% of the original proportion.

            eligible_categories = [
                category
                for category, proportion in feature.items()
                if proportion > -diff and abs(diff) <= (proportion * 0.1)
            ]
            if not len(eligible_categories):
                raise ValueError(
                    "No eligible category found for rebalancing: failed to round target proportions total to 1."
                )
            # Choosing a random key to change so that we don't over bias one category more than others
            # Note: this means the results will be stochastic
            lsoa_marginals[feature_name][random.choice(eligible_categories)] += diff
            if not round(sum(feature.values()), round_n) == 1:
                raise AssertionError(
                    f"Rounding proportions not successful. Sum of proportions is {sum(feature.values())}, not 1"
                )

    return lsoa_marginals
