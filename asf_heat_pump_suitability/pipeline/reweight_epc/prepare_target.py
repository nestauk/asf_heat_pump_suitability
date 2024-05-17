import balance
from balance.weighting_methods import rake
import polars as pl
import polars.selectors as cs
from typing import Dict
from asf_heat_pump_suitability.getters import get_target


def generate_balance_target_population(
    target_marginals: Dict[str, dict], lsoa: str
) -> balance.sample_class.Sample:
    """
    Generate balance Sample object from target proportions for each feature category for the specified LSOA.

    Args:
        target_marginals (Dict[str, dict]): dict of dicts containing target proportions for each feature category by LSOA
        lsoa (str): LSOA to generate target population for

    Returns:
        balance.sample_class.Sample: artificial target population object generated from given marginals
    """
    target = {k: v[lsoa] for k, v in target_marginals.items()}
    df = rake.prepare_marginal_dist_for_raking(target)
    return balance.Sample.from_frame(df)


def get_dict_target_marginals() -> Dict[str, pl.DataFrame]:
    """
    Get dict of dicts containing target proportions of each feature category per LSOA. Dictionary keys are
    feature names.

    Returns:
        Dict[str, dict]: dict of dicts containing target proportions for each feature category per LSOA
    """
    target_features = get_dict_dfs_counts()

    target_proportions = {
        k: convert_df_proportions(v) for k, v in target_features.items()
    }

    return {k: to_dict_feature_marginals(v) for k, v in target_proportions.items()}


def get_dict_dfs_counts() -> Dict[str, pl.DataFrame]:
    """
    Get dict of dataframes containing counts of each feature category per LSOA in the target datasets. Dictionary
    keys are feature names.

    Returns:
        Dict[str, pl.DataFrame]: dict of dataframes containing counts of each feature category per LSOA in the target
        datasets
    """
    return {
        "tenure": get_target.get_df_target_tenure(),
        "property_type": get_target.get_df_target_property_type(),
        # TODO: adding nrooms feature causes crash when running `generate_balance_target_population`
        # "nrooms": get_target.get_df_target_nrooms(),
        "build_year": get_target.get_df_target_build_year(),
    }


def convert_df_proportions(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """
    Convert dataframe of counts to target proportions for each feature category per LSOA.

    Args:
        pl.DataFrame: dataframe containing counts of each feature category per LSOA

    Returns:
        Dict[str, pl.DataFrame]: dataframe of target proportions for each feature category per LSOA
    """
    cols = df.select(cs.integer()).columns
    df = df.with_columns(pl.sum_horizontal(df.select(cols)).alias("total"))
    df = df.with_columns(
        [(pl.col(col) / pl.col("total")).alias(col) for col in cols]
    ).drop(columns=["total"])

    return df


def to_dict_feature_marginals(df: pl.DataFrame) -> Dict[str, dict]:
    """
    Convert dataframe with target marginals to dict where keys are LSOA codes

    Args:
        df (pl.DataFrame): dataframe with target marginals category per LSOA

    Returns:
        Dict[str, dict]: dict with target proportions for each feature category where keys are LSOA codes
    """
    lsoas = df["lsoa"].to_list()
    marginals = df.select(pl.exclude("lsoa")).to_dicts()

    return dict(zip(lsoas, marginals))
