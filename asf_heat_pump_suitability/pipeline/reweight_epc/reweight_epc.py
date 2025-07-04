"""
Functions to reweight EPC using IPF against target marginals.
"""

import polars as pl
import random
from typing import Dict, Tuple
import balance
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target


def get_dict_sample_weights(
    weighted_sample: balance.sample_class.Sample,
) -> Dict[str, list]:
    """
    Get dictionary mapping containing UPRNs and weights of weighted EPC sample rows.

    Args:
        weighted_sample (balance.sample_class.Sample): EPC data with weights adjusted to target marginals for a single LSOA

    Returns:
        Dict[str, list]: dictionary mapping column names to list of UPRNs and corresponding weights
    """
    weighted_sample = pl.from_pandas(weighted_sample.df)
    weighted_sample = weighted_sample.select(["UPRN", "weight"]).with_columns(
        (pl.col("weight") / pl.col("weight").sum()).alias("proportional_weight")
    )  # convert to proportional weight
    weights_dict = weighted_sample.to_dict(as_series=False)

    return weights_dict


def generate_weighted_sample(
    balance_sample: balance.sample_class.Sample,
    balance_target: balance.sample_class.Sample,
) -> balance.sample_class.Sample:
    """
    Generate weighted EPC sample for single LSOA using target marginals.

    Args:
        balance_sample (balance.sample_class.Sample): EPC data object for a single LSOA prepared for weighting
        balance_target (balance.sample_class.Sample): artificial target population object generated from given marginals for single LSOA

    Returns:
        balance.sample_class.Sample: EPC data with weights adjusted to target marginals for single LSOA
    """
    sample_w_target = balance_sample.set_target(balance_target)
    weighted_sample = sample_w_target.adjust(method="rake")
    assert weighted_sample.is_adjusted()

    return weighted_sample


def generate_balance_sample(
    df: pl.DataFrame, features: list, target_marginals: Dict[str, dict], lsoa: str
) -> Tuple[balance.sample_class.Sample, int]:
    """
    Prepare EPC subset for weighting with `balance` package for the single specified LSOA and identify the number of
    rows lost during preprocessing for weighting. Preprocessing adds dummy rows with dummy data for any feature
    categories present in target but not in EPC sample, and drops rows with nulls in any weighting features.

    Args:
        df (pl.DataFrame): EPC dataset
        features (list): names of feature columns in EPC dataset to use in weighting
        target_marginals (Dict[str, dict]): dict of dicts containing target proportions for each feature category by LSOA
        lsoa (str): code of LSOA to weight EPC data for

    Returns:
        Tuple[balance.sample_class.Sample, int]: EPC data sample for a single LSOA prepared for weighting with number of
        rows lost during preprocessing. Returns `None` in place of EPC data sample if all EPC rows are lost during preprocessing.
    """
    # Get EPC sample filtered for single given LSOA
    cols = ["lsoa", "UPRN"]
    cols.extend(features)
    sample = df.select(cols).filter(pl.col("lsoa") == lsoa)

    # Prepare target marginals for single LSOA
    lsoa_marginals = prepare_target.get_dict_lsoa_marginals(
        target_marginals=target_marginals, lsoa=lsoa
    )

    # Remove rows from EPC sample where feature category is not present in target dataset
    len_before = len(sample)
    for feature, marginals in lsoa_marginals.items():
        missing = _generate_list_missing_from_target(
            feature=feature, feature_marginals=marginals, sample=sample
        )
        sample = sample.filter(~pl.col(feature).is_in(missing))
    lost_rows = len_before - len(sample)

    if len(sample) > 0:
        # Add dummy rows for feature categories missing from sample but present in target
        dummies = generate_df_dummies(lsoa_marginals=lsoa_marginals, sample=sample)
        sample = pl.concat([sample, dummies[sample.columns]])

        # Convert to balance sample
        cols.remove("lsoa")
        balance_sample = sample.to_pandas()
        balance_sample = balance.Sample.from_frame(
            balance_sample[cols], id_column="UPRN"
        )

        return balance_sample, lost_rows

    else:
        return None, lost_rows


def generate_df_dummies(
    sample: pl.DataFrame, lsoa_marginals: Dict[str, dict]
) -> pl.DataFrame:
    """
    Generate dummy rows to add to EPC subset for single LSOA. Dummy rows contain feature categories present in target
    dataset but missing from EPC subset. Dummy rows are required when EPC subset is missing target feature categories to
    improve the weighting output.

    Args:
        sample (pl.DataFrame): EPC dataset subset to a single LSOA
        lsoa_marginals (Dict[str, dict]): dict of dicts containing target proportions for each feature category for a
        single LSOA

    Returns:
        pl.DataFrame: dummy rows for a single LSOA in the EPC dataset
    """
    dfs = []

    # Get feature categories missing from EPC sample for each feature
    for feature, marginals in lsoa_marginals.items():
        missing = _generate_list_missing_from_sample(
            feature=feature, feature_marginals=marginals, sample=sample
        )
        dfs.append(pl.DataFrame({feature: missing}))

    # Generate dummy rows with missing categories
    dummy_rows = pl.concat(dfs, how="horizontal").with_columns(
        pl.lit(sample["lsoa"].unique()[0], pl.String).alias("lsoa")
    )

    # Add id to dummy rows and fill nulls in dummy rows
    dummy_rows = dummy_rows.with_columns(
        pl.Series(name="UPRN", values=[f"dummy_{_}" for _ in range(0, len(dummy_rows))])
    )
    dummy_rows = _fill_nulls_dummy_df(
        dummy_rows=dummy_rows, lsoa_marginals=lsoa_marginals, sample=sample
    )

    return dummy_rows


def _fill_nulls_dummy_df(
    dummy_rows: pl.DataFrame, sample: pl.DataFrame, lsoa_marginals: Dict[str, dict]
) -> pl.DataFrame:
    """
    Fill nulls in dummy rows for each feature with another randomly selected missing category if available. If no categories
    are missing for a feature, fill with the most common category in the EPC subset.

    Args:
        dummy_rows (pl.DataFrame): dummy rows containing missing feature categories for a single LSOA in the EPC dataset
        sample (pl.DataFrame): EPC dataset subset to a single LSOA
        lsoa_marginals (Dict[str, dict]): dict of dicts containing target proportions for each feature category for a
        single LSOA

    Returns:
        pl.DataFrame: complete dummy rows for a single LSOA in the EPC dataset
    """
    for feature, marginals in lsoa_marginals.items():
        if dummy_rows[feature].is_null().any():
            # Identify missing categories for each feature
            missing = _generate_list_missing_from_sample(
                feature=feature, feature_marginals=marginals, sample=sample
            )

            # Fill nulls with random missing category from feature if exists
            if len(missing) > 0:
                dummy_rows = dummy_rows.with_columns(
                    pl.col(feature).fill_null(random.choices(missing, k=1)[0])
                )
            # Otherwise, fill nulls with category with max count in sample
            else:
                dummy_rows = dummy_rows.with_columns(
                    pl.col(feature).fill_null(
                        sample[feature].value_counts().max()[feature][0]
                    )
                )

    return dummy_rows


def _generate_list_missing_from_sample(
    sample: pl.DataFrame, feature: str, feature_marginals: Dict[str, float]
) -> list:
    """
    Generate list of categories present in target data but missing from EPC subset for the specified feature.

    Args:
        sample (pl.DataFrame): EPC dataset subset to a single LSOA
        feature (str): name of feature
        feature_marginals (Dict[str, float]): dict containing target proportions for one feature for a single LSOA

    Returns:
        list: categories missing from EPC subset for a given feature
    """
    sample_cats = set(sample[feature].unique())
    target_cats = {k for k in feature_marginals.keys() if feature_marginals.get(k) > 0}
    missing = list(target_cats.difference(sample_cats))
    return missing


def _generate_list_missing_from_target(
    sample: pl.DataFrame, feature: str, feature_marginals: Dict[str, float]
) -> list:
    """
    Generate list of categories present in EPC subset but missing from target data for the specified feature.

    Args:
        sample (pl.DataFrame): EPC dataset subset to a single LSOA
        feature (str): name of feature
        feature_marginals (Dict[str, float]): dict containing target proportions for one feature for a single LSOA

    Returns:
        list: categories missing from target for a given feature
    """
    sample_cats = set(sample[feature].unique())
    target_cats = {k for k in feature_marginals.keys() if feature_marginals.get(k) > 0}
    missing = list(sample_cats.difference(target_cats))
    return missing
