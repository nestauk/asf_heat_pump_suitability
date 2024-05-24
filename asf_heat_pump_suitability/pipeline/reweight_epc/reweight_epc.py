import polars as pl
import random
from typing import Dict, Tuple
import balance
import logging
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target


def get_tuple_sample_weights(
    reweighted_sample: balance.sample_class.Sample,
) -> Tuple[list, list]:
    """
    Get Tuple of lists containing UPRNs and weights of reweighted EPC sample rows.

    Args:
        reweighted_sample (balance.sample_class.Sample): EPC data object with weights adjusted to target marginals for single LSOA

    Returns:
        Tuple[list, list]: UPRNs and corresponding weights
    """
    reweighted_sample = pl.from_pandas(reweighted_sample.df)
    df = (
        reweighted_sample.select(["UPRN", "weight"])
        .filter(~pl.col("UPRN").str.contains("dummy"))
        .with_columns((pl.col("weight") / pl.col("weight").sum()).alias("weight"))
    )

    uprns = df.to_dict()["UPRN"].to_list()
    weights = df.to_dict()["weight"].to_list()

    return uprns, weights


def generate_reweighted_sample(
    balance_sample: balance.sample_class.Sample,
    balance_target: balance.sample_class.Sample,
) -> balance.sample_class.Sample:
    """
    Generate reweighted EPC subset object for single LSOA using target marginals.

    Args:
        balance_sample (balance.sample_class.Sample): EPC data object for a single LSOA prepared for reweighting
        balance_target (balance.sample_class.Sample): artificial target population object generated from given marginals for single LSOA

    Returns:
        balance.sample_class.Sample: EPC data object with weights adjusted to target marginals for single LSOA
    """
    sample_w_target = balance_sample.set_target(balance_target)
    adjusted_sample = sample_w_target.adjust(method="rake")
    assert adjusted_sample.is_adjusted()

    return adjusted_sample


def generate_balance_sample(
    df: pl.DataFrame, features: list, target_marginals: Dict[str, dict], lsoa: str
) -> Tuple[balance.sample_class.Sample, int]:
    """
    Prepare EPC subset for reweighting with `balance` package for the single specified LSOA.

    Args:
        df (pl.DataFrame): EPC dataset
        features (list): names of feature columns in EPC dataset
        target_marginals (Dict[str, dict]): dict of dicts containing target proportions for each feature category by LSOA
        lsoa (str): LSOA to reweight

    Returns:
        Tuple[balance.sample_class.Sample, int]: EPC data object for a single LSOA prepared for reweighting
    """
    cols = ["lsoa", "UPRN"]
    cols.extend(features)
    sample = df.select(cols).filter(pl.col("lsoa") == lsoa)
    lsoa_marginals = prepare_target.get_dict_lsoa_marginals(
        target_marginals=target_marginals, lsoa=lsoa
    )
    len_before = len(sample)
    for feature, marginals in lsoa_marginals.items():
        missing = _generate_list_missing_from_target(
            feature=feature, feature_marginals=marginals, sample=sample
        )
        sample = sample.filter(~pl.col(feature).is_in(missing))
    lost_rows = len(sample) - len_before
    dummies = generate_df_dummies(lsoa_marginals=lsoa_marginals, sample=sample)
    sample = pl.concat([sample, dummies[sample.columns]])
    cols.remove("lsoa")
    balance_sample = sample.to_pandas()
    balance_sample = balance.Sample.from_frame(balance_sample[cols], id_column="UPRN")

    return balance_sample, lost_rows


def drop_nulls_feature_cols(df, features):
    """
    Drop rows with null values in any feature column from EPC dataset.

    Args:
        df (pl.DataFrame): EPC dataset
        features (list): column names of features

    Returns:
        pl.DataFrame: EPC dataset where rows with null in any specified feature column are dropped
    """
    df = df.with_columns(
        pl.col(["tenure", "property_type"]).replace(
            {"unknown": None}, return_dtype=pl.String
        )
    ).drop_nulls(subset=features)
    return df


def add_cols_weighting_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add standardised feature columns to be used for weighting, to EPC dataset.

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with standardised feature columns
    """
    df = add_col_property_type(df)
    # df = add_col_nrooms(df)
    df = add_col_build_year_1930(df)
    df = df.rename({"TENURE": "tenure"})
    return df


def add_col_property_type(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `property_type` column to EPC dataset with property type categories corresponding to those from the census.

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with `property_type` column
    """
    terraced = [
        "Mid-Terrace",
        "End-Terrace",
        "Enclosed Mid-Terrace",
        "Enclosed End-Terrace",
    ]

    df = df.with_columns(
        pl.when(
            pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]),
            pl.col("BUILT_FORM") == "Detached",
        )
        .then(pl.lit("Detached whole house or bungalow"))
        .when(
            pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]),
            pl.col("BUILT_FORM") == "Semi-Detached",
        )
        .then(pl.lit("Semi-detached whole house or bungalow"))
        .when(
            pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]),
            pl.col("BUILT_FORM").is_in(terraced),
        )
        .then(pl.lit("Terraced (including end-terrace) whole house or bungalow"))
        .when(pl.col("PROPERTY_TYPE").is_in(["Flat", "Maisonette"]))
        .then(pl.lit("Flat, maisonette or apartment"))
        .when(pl.col("PROPERTY_TYPE").is_in(["Park home"]))
        .then(pl.lit("A caravan or other mobile or temporary structure"))
        .alias("property_type")
    )

    return df


def add_col_nrooms(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `number_of_rooms` column to EPC dataset with categories corresponding to those from the census.

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with `number_of_rooms` column
    """
    return df.with_columns(
        pl.col("NUMBER_HABITABLE_ROOMS")
        .map_elements(lambda x: 9 if x > 9 else x, return_dtype=pl.Float32)
        .cast(pl.Int8)
        .alias("number_of_rooms")
    )


def add_col_build_year_1930(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `build_year` column to EPC dataset indicating whether property construction year is pre- or post-1930 (or unknown).

    Args:
        df (pl.DataFrame): EPC dataset

    Returns:
        pl.DataFrame: EPC dataset with `build_year` column
    """
    return df.with_columns(
        pl.col("CONSTRUCTION_AGE_BAND")
        .map_dict(config["mapping"]["pre_post_1930_epc"])
        .alias("build_year")
    )


def generate_df_dummies(
    sample: pl.DataFrame, lsoa_marginals: Dict[str, dict]
) -> pl.DataFrame:
    """
    Generate dummy rows to add to EPC subset for single LSOA. Dummy rows contain feature categories present in target
    dataset but missing from EPC subset. Dummy rows are required when EPC subset is missing categories to improve
    the reweighting output.

    Args:
        sample (pl.DataFrame): EPC dataset subset to a single LSOA
        lsoa_marginals (Dict[str, dict]): dict of dicts containing target proportions for each feature category for a
        single LSOA

    Returns:
        pl.DataFrame: dummy rows for a single LSOA in the EPC dataset
    """
    _dfs = []
    for feature, marginals in lsoa_marginals.items():
        missing = _generate_list_missing_categories(
            feature=feature, feature_marginals=marginals, sample=sample
        )
        _dfs.append(pl.DataFrame({feature: missing}))

    dummy_rows = pl.concat(_dfs, how="horizontal").with_columns(
        pl.lit(sample["lsoa"].unique()[0], pl.String).alias("lsoa")
    )

    # Add id to dummy rows
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
            missing = _generate_list_missing_categories(
                feature=feature, feature_marginals=marginals, sample=sample
            )
            if len(missing) > 0:  # fill with random missing category from feature first
                dummy_rows = dummy_rows.with_columns(
                    pl.col(feature).fill_null(random.choices(missing, k=1)[0])
                )
            else:  # otherwise, fill with category with max count in sample
                dummy_rows = dummy_rows.with_columns(
                    pl.col(feature).fill_null(
                        sample[feature].value_counts().max()[feature][0]
                    )
                )
    return dummy_rows


def _generate_list_missing_categories(
    feature: str, feature_marginals: Dict[str, float], sample: pl.DataFrame
) -> list:
    """
    Generate list of categories present in target data but missing from EPC subset for the specified feature.

    Args:
        feature (str): name of feature
        feature_marginals (Dict[str, float]): dict containing target proportions for one feature for a single LSOA
        sample (pl.DataFrame): EPC dataset subset to a single LSOA

    Returns:
        list: categories missing from EPC subset for a given feature
    """
    cats = {k for k in feature_marginals.keys() if feature_marginals.get(k) > 0}
    missing = list(cats.difference(set(sample[feature].unique())))
    return missing


def _generate_list_missing_from_target(
    feature: str, feature_marginals: Dict[str, float], sample: pl.DataFrame
) -> list:
    """
    Generate list of categories present in target data but missing from EPC subset for the specified feature.

    Args:
        feature (str): name of feature
        feature_marginals (Dict[str, float]): dict containing target proportions for one feature for a single LSOA
        sample (pl.DataFrame): EPC dataset subset to a single LSOA

    Returns:
        list: categories missing from EPC subset for a given feature
    """
    sample_cats = set(sample[feature].unique())
    target_cats = {k for k in feature_marginals.keys() if feature_marginals.get(k) > 0}
    missing = list(sample_cats.difference(target_cats))
    return missing
