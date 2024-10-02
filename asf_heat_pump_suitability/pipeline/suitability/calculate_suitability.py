"""
Functions to calculate suitability of different HP technologies.
"""

import polars as pl
import s3fs
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from asf_heat_pump_suitability import config


site_regs_scores = {
    "ASHP_S": 0.25,
    "ASHP_N": 0.25,
    "GSHP_S": 0.25,
    "GSHP_N": 0.25,
    "SGL_S": 0.25,
    "SGL_N": 0.25,
    "HN_S": 0.25,
    "HN_N": 0.25,
}

grid_capacity_scores = {
    "ASHP_S": 1,
    "ASHP_N": 0,
    "GSHP_S": 1,
    "GSHP_N": 0,
    "SGL_S": 1,
    "SGL_N": 0,
    "HN_S": 0,
    "HN_N": 0,
}

epc_threshold_scores = {
    "ASHP_S": 1,
    "ASHP_N": 0,
    "GSHP_S": 1,
    "GSHP_N": 0,
    "SGL_S": 1,
    "SGL_N": 0,
    "HN_S": 0,
    "HN_N": 0,
}

water_tank_space_scores = {
    "ASHP_S": 1,
    "ASHP_N": 1,
    "GSHP_S": 1,
    "GSHP_N": 1,
    "SGL_S": 1,
    "SGL_N": 1,
    "HN_S": 1,
    "HN_N": 1,
}

garden_size_scores = {
    "ASHP_S": 1,
    "ASHP_N": 0,
    "GSHP_S": 1,
    "GSHP_N": 0,
    "SGL_S": 1,
    "SGL_N": 0,
    "HN_S": 0,
    "HN_N": 0,
}

external_space_scores = {
    "ASHP_S": 0,
    "ASHP_N": 2,
    "GSHP_S": 0,
    "GSHP_N": 1,
    "SGL_S": 0,
    "SGL_N": 0,
    "HN_S": 0,
    "HN_N": 0,
}

offgas_scores = {
    "ASHP_S": 0.5,
    "ASHP_N": 0.5,
    "GSHP_S": 0.5,
    "GSHP_N": 0.5,
    "SGL_S": 0.5,
    "SGL_N": 0.5,
    "HN_S": 0.5,
    "HN_N": 0.5,
}

property_density_scores = {
    "ASHP_S": 0,
    "ASHP_N": 0,
    "GSHP_S": 0,
    "GSHP_N": 0,
    "SGL_S": 2,
    "SGL_N": 2,
    "HN_S": 0,
    "HN_N": 0,
}

high_heat_demand_scores = {
    "ASHP_S": 0,
    "ASHP_N": 0,
    "GSHP_S": 0,
    "GSHP_N": 0,
    "SGL_S": 0,
    "SGL_N": 0,
    "HN_S": 2,
    "HN_N": 2,
}

anchor_properties_scores = {
    "ASHP_S": 0,
    "ASHP_N": 0,
    "GSHP_S": 0,
    "GSHP_N": 0,
    "SGL_S": 0,
    "SGL_N": 0,
    "HN_S": 1,
    "HN_N": 1,
}

not_flat_scores = {
    "ASHP_S": 1,
    "ASHP_N": 1,
    "GSHP_S": 1,
    "GSHP_N": 1,
    "SGL_S": 0,
    "SGL_N": 0,
    "HN_S": 0,
    "HN_N": 0,
}

multiple_props_scores = {
    "ASHP_S": 0,
    "ASHP_N": 0,
    "GSHP_S": 0,
    "GSHP_N": 0,
    "SGL_S": 2,
    "SGL_N": 2,
    "HN_S": 2,
    "HN_N": 2,
}

# to refine
grid_capacity_scores = {
    "ASHP_S": 1,
    "ASHP_N": 1,
    "GSHP_S": 1,
    "GSHP_N": 1,
    "SGL_S": 1,
    "SGL_N": 1,
    "HN_S": 1,
    "HN_N": 1,
}


def parse_arguments():
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epc_path",
        help="Path to parquet with EPC properties with added features and weights for calculating suitability score from.",
        required=True,
    )

    return parser.parse_args()


def get_enhanced_epc(path) -> pl.DataFrame:
    """
    Load EPC dataset enhanced with weights and additional features.

    Returns:
        pl.DataFrame: enhanced EPC dataset
    """
    usecols = [
        "UPRN",
        "COUNTRY",
        "lsoa",
        "weight",
        "proportional_weight",
        "ruc_two_fold",
        "OFF GAS",
        "Property density (households per KM2)",
        "garden_area_m2",
        "listed_building_grade",
        "in_conservation_area",
        "lad_conservation_area_data_available",
        "property_type",
        "CURRENT_ENERGY_RATING",
        "has_grid_capacity",
    ]
    df = pl.read_parquet(path, columns=usecols)

    df = df.filter(
        ~pl.col("UPRN").str.contains("dummy"), pl.col("COUNTRY") != "Scotland"
    )

    df = df.with_columns(
        pl.when(pl.col("listed_building_grade").is_null())
        .then(False)
        .otherwise(True)
        .alias("listed_building"),
    )

    return df


def compute_df_avg_score_per_epc(
    df: pl.DataFrame,
    tech_type: str,
    density_threshold: int = 100,
    garden_threshold: int = 10,
    external_space_threshold: int = 2,
):
    """
    Calculate average heat pump suitability score per EPC record for specified tech type.

    Args:
        df: EPC dataset with features for calculating suitability score
        tech_type (str): tech type to calculate suitability scores for, in standard (S) or Nesta (N) view
        density_threshold: minimum property density (households per km2) required for shared ground loop
        garden_threshold: minimum garden size (m2) required for heat pumps
        external_space_threshold: minimum outdoor space (m2) required for heat pumps

    Returns:
        pl.DataFrame: average suitability score for specified tech type
    """
    scores_df = compute_df_total_score_per_epc(
        df, tech_type, density_threshold, garden_threshold, external_space_threshold
    )
    max_scores_df = compute_df_max_score_per_row(df, tech_type)
    df = scores_df.join(max_scores_df, on="UPRN", how="inner")
    df = df.with_columns(
        (pl.col(f"{tech_type}_score") / pl.col(f"{tech_type}_max_score")).alias(
            f"{tech_type}_avg_score"
        )
    )
    return df.select(["UPRN", f"{tech_type}_avg_score"])


def compute_df_total_score_per_epc(
    df: pl.DataFrame,
    tech_type: str,
    density_threshold: int = 60,
    garden_threshold: int = 10,
    external_space_threshold: int = 2,
) -> pl.DataFrame:
    """
    Calculate total heat pump suitability score points per EPC record for specified tech_type.

    Args:
        df: EPC dataset with features for calculating suitability score
        tech_type (str): tech type to calculate suitability scores for, in standard (S) or Nesta (N) view
        density_threshold: minimum property density (households per km2) required for shared ground loop
        garden_threshold: minimum garden size (m2) required for heat pumps
        external_space_threshold: minimum outdoor space (m2) required for heat pumps

    Returns:
        pl.DataFrame: suitability score for specified tech type
    """
    df = df.with_columns(
        pl.when(pl.col("ruc_two_fold") == "Urban")
        .then(high_heat_demand_scores.get(tech_type))
        .alias("heat_demand_score"),
        pl.when(pl.col("OFF GAS"))
        .then(offgas_scores.get(tech_type))
        .alias("off_gas_score"),
        pl.when(pl.col("Property density (households per KM2)") > density_threshold)
        .then(property_density_scores.get(tech_type))
        .alias("property_density_score"),
        pl.when(pl.col("garden_area_m2") > garden_threshold)
        .then(garden_size_scores.get(tech_type))
        .alias("garden_size_score"),
        pl.when(pl.col("garden_area_m2") > external_space_threshold)
        .then(external_space_scores.get(tech_type))
        .alias("external_space_score"),
        pl.when(~pl.col("listed_building"))
        .then(site_regs_scores.get(tech_type))
        .alias("not_listed_score"),
        pl.when(~pl.col("in_conservation_area"))
        .then(site_regs_scores.get(tech_type))
        .alias("not_in_cons_area_score"),
        pl.when(pl.col("property_type") != "Flat, maisonette or apartment")
        .then(not_flat_scores.get(tech_type))
        .otherwise(multiple_props_scores.get(tech_type))
        .alias("property_type_score"),
        pl.when(pl.col("CURRENT_ENERGY_RATING").is_in(["A", "B", "C"]))
        .then(epc_threshold_scores.get(tech_type))
        .alias("epc_rating_score"),
        pl.when(pl.col("has_grid_capacity"))
        .then(epc_threshold_scores.get(tech_type))
        .alias("grid_capacity_score"),
    )

    score_cols = [col for col in df.columns if "score" in col]
    df = df.with_columns(pl.sum_horizontal(score_cols).alias(f"{tech_type}_score"))

    return df.select(["UPRN", f"{tech_type}_score"])


def compute_df_max_score_per_row(df: pl.DataFrame, tech_type: str) -> pl.DataFrame:
    """
    Get max score possible per row. This is calculated by adding together the scores for the tech type for each feature
    in the row that is not null.

    Args:
        df: EPC dataset with features for calculating suitability score
        tech_type (str): tech type to calculate suitability scores for, in standard (S) or Nesta (N) view

    Returns:
        pl.DataFrame: max possible score per EPC row
    """
    df = df.with_columns(
        pl.when(pl.col("ruc_two_fold").is_not_null())
        .then(high_heat_demand_scores.get(tech_type))
        .otherwise(0)
        .alias("heat_demand_max"),
        pl.when(pl.col("OFF GAS").is_not_null())
        .then(offgas_scores.get(tech_type))
        .otherwise(0)
        .alias("off_gas_max"),
        pl.when(pl.col("Property density (households per KM2)").is_not_null())
        .then(property_density_scores.get(tech_type))
        .otherwise(0)
        .alias("property_density_max"),
        pl.when(pl.col("garden_area_m2").is_not_null())
        .then(garden_size_scores.get(tech_type) + external_space_scores.get(tech_type))
        .otherwise(0)
        .alias("garden_size_max"),
        pl.when(pl.col("listed_building").is_not_null())
        .then(site_regs_scores.get(tech_type))
        .alias("listed_buildings_max"),
        pl.when(pl.col("in_conservation_area").is_not_null())
        .then(site_regs_scores.get(tech_type))
        .alias("in_cons_area_max"),
        pl.when(pl.col("property_type").is_not_null())
        .then(
            # Note: this only works because not_flat and multiple_props scores are mutually exclusive
            (not_flat_scores.get(tech_type) + multiple_props_scores.get(tech_type))
        )
        .otherwise(0)
        .alias("property_type_max"),
        pl.when(pl.col("CURRENT_ENERGY_RATING").is_not_null())
        .then(epc_threshold_scores.get(tech_type))
        .otherwise(0)
        .alias("epc_rating_max"),
        pl.when(pl.col("has_grid_capacity").is_not_null())
        .then(epc_threshold_scores.get(tech_type))
        .otherwise(0)
        .alias("grid_capacity_max"),
    )

    max_cols = [col for col in df.columns if "max" in col]
    df = df.with_columns(pl.sum_horizontal(max_cols).alias(f"{tech_type}_max_score"))

    return df.select(["UPRN", f"{tech_type}_max_score"])


def filter_df_minimum_features(
    df: pl.DataFrame, features: list = None, threshold: int = 4
) -> pl.DataFrame:
    """
    Calculate number of non-null features for each row of EPC.

    Args:
        df: EPC dataset with features.
        features: list of features to calculate HP suitability
        threshold: minimum features required to be included
    """
    if features is None:
        features = [
            "ruc_two_fold",
            "OFF GAS",
            "Property density (households per KM2)",
            "garden_area_m2",
            "listed_building_grade",
            "in_conservation_area",
            "property_type",
            "CURRENT_ENERGY_RATING",
            "has_grid_capacity",
        ]
    df = df.with_columns(
        (len(features) - pl.sum_horizontal(pl.col(features).is_null()))
    ).rename(
        {"literal": "n_features"}
    )  # Not sure why this is required, naming with alias directly doesnt work fsr

    df = df.filter(pl.col("n_features") >= threshold)

    return df


def compute_df_weighted_score(df, threshold=0.5):
    """
    Calculate [un]weighted suitability scores per EPC property in a single LSOA. Scores will only be weighted if the
    proportion of EPC properties in the LSOA with non-null weight data is above the specified threshold.

    Args:
        df: EPC dataset for one LSOA with suitability scores per property for each tech type and with proportional weights
        threshold (float): minimum proportion of properties in LSOA EPC sample with non-null weights

    Returns:
        pl.DataFrame: weighted scores for an LSOA
    """
    score_cols = [col for col in df.columns if "score" in col]
    df = df.with_columns(
        pl.when(
            (pl.col("proportional_weight").is_not_null().sum() / len(df)) >= threshold
        )
        .then(pl.col("proportional_weight") / pl.col("proportional_weight").sum())
        .otherwise(1)
        .alias("use_weight"),
        pl.when(
            (pl.col("proportional_weight").is_not_null().sum() / len(df)) >= threshold
        )
        .then(True)
        .otherwise(False)
        .alias("scores_weighted"),
    )
    for col in score_cols:
        df = df.with_columns(
            (pl.col(col) * pl.col("use_weight")).alias(f"{col}_weighted")
        )

    return df


def compute_dict_lsoa_suitability_scores(df: pl.DataFrame, lsoa: str) -> dict:
    """
    Calculate average heat pump suitability scores for LSOA per tech type.

    Args:
        df (pl.DataFrame): LSOA with weighted suitability scores per tech type
        lsoa (str): LSOA code

    Returns:
        dict: suitability scores per LSOA for each tech type
    """
    scores_dict = {"lsoa": lsoa}
    assert df["scores_weighted"].n_unique() == 1
    score_cols = [col for col in df.columns if "score_weighted" in col]
    for score in score_cols:
        if df["scores_weighted"].unique()[0]:
            scores_dict[score] = df[score].sum()
        else:
            scores_dict[score] = df[score].mean()
    scores_dict["scores_weighted"] = df["scores_weighted"].unique()[0]
    scores_dict["n_properties"] = len(df)

    return scores_dict


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    # TODO: logging.info not displaying to terminal for me
    logger.info("Loading EPC data with features")
    epc_df = get_enhanced_epc(path=args.epc_path)

    logger.info("Filtering EPC data to rows with n_features >= minimum threshold")
    epc_df = filter_df_minimum_features(epc_df)

    tech_types = [
        "ASHP_S",
        "ASHP_N",
        "GSHP_S",
        "GSHP_N",
        "SGL_S",
        "SGL_N",
        "HN_S",
        "HN_N",
    ]

    scores = []
    for tech_type in tech_types:
        logger.info(f"Calculating suitability scores for tech type: {tech_type}")
        epc_scores_df = compute_df_avg_score_per_epc(epc_df, tech_type)
        scores.append(epc_scores_df)

    logger.info("Joining all scores to EPC dataset")
    for score_df in scores:
        epc_df = epc_df.join(score_df, on="UPRN", how="left")

    fs = s3fs.S3FileSystem()
    save_as = f"s3://asf-heat-pump-suitability/outputs/2023Q4/{datetime.today().strftime('%Y%m%d')}_2023_Q4_heat_pump_suitability_per_property.parquet"
    with fs.open(save_as, mode="wb") as f:
        epc_df.write_parquet(f)

    logger.info("Weighting scores and aggregating per LSOA")
    weighted_scores = []
    for lsoa_code in tqdm(epc_df["lsoa"].unique()):
        lsoa_df = epc_df.filter(pl.col("lsoa") == lsoa_code)
        lsoa_df = compute_df_weighted_score(lsoa_df)
        weighted_scores.append(compute_dict_lsoa_suitability_scores(lsoa_df, lsoa_code))
    # Must have at least 15 properties to be included in score
    suitability_df = pl.DataFrame(weighted_scores).filter(pl.col("n_properties") >= 15)
    suitability_df = suitability_df.with_columns(pl.col(pl.Float64).round(3))

    logger.info("Get LSOA names and join to suitability dataset")
    lsoa_names_df = pl.read_csv(
        config["data_source"]["EW_ons_lsoa_lad_lookup"],
        columns=["LSOA21CD", "LSOA21NM"],
    )
    suitability_df = suitability_df.join(
        lsoa_names_df, left_on="lsoa", right_on="LSOA21CD", how="left"
    ).rename({"LSOA21NM": "lsoa_name"})

    logger.info("Saving LSOA heat pump suitability scores")
    fs = s3fs.S3FileSystem()
    save_as = f"s3://asf-heat-pump-suitability/outputs/2023Q4/{datetime.today().strftime('%Y%m%d')}_2023_Q4_heat_pump_suitability_per_lsoa"
    with fs.open(f"{save_as}.parquet", mode="wb") as f:
        suitability_df.write_parquet(f)
    with fs.open(f"{save_as}.csv", mode="wb") as f:
        suitability_df.write_csv(f)
