"""
Functions to calculate suitability of different HP technologies.
"""

import polars as pl
from collections import defaultdict
import logging
import s3fs
from datetime import datetime
from tqdm import tqdm
from typing import Optional

logging.getLogger().setLevel(logging.INFO)

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


def get_enhanced_epc() -> pl.DataFrame:
    """
    Load EPC dataset enhanced with weights and additional features.

    Returns:
        pl.DataFrame: enhanced EPC dataset
    """
    usecols = [
        "UPRN",
        "lsoa",
        "weight",
        "proportional_weight",
        "ruc_two_fold",
        "OFF GAS",
        "Property density (households per KM2)",
        "msoa_avg_outdoor_space_m2",
        "listed_building_grade",
        "in_conservation_area",
        "lad_conservation_area_data_available",
        "property_type",
        "CURRENT_ENERGY_RATING",
    ]
    df = pl.read_parquet(
        "s3://asf-heat-pump-suitability/outputs/20240827_2023_Q4_EPC_weighted_features.parquet",
        columns=usecols,
    )

    df = df.filter(~pl.col("UPRN").str.contains("dummy"))

    df = df.with_columns(
        pl.when(pl.col("listed_building_grade").is_null())
        .then(False)
        .otherwise(True)
        .alias("listed_building"),
    )

    return df


def score_dict_site_regs(
    tech_type, conservation_zone, listed_building
) -> Optional[dict]:
    """
    Is this property NOT listed / in a conservation zone / any other planning regulations?
    """
    if conservation_zone or listed_building:
        return None
    else:
        return site_regs_scores.get(tech_type)


def score_dict_grid_capacity(tech_type, lsoa):
    """
    Does the electricity grid for the LSOA this property is in have capacity?
    """
    capacity_per_lsoa = {22: True, 55: False}  # To define

    if capacity_per_lsoa.get(lsoa):
        return grid_capacity_scores.get(tech_type)
    else:
        return None


def score_dict_epc_rating(tech_type, epc):
    """
    Is the EPC rating >= C?
    """

    if epc in ["A", "B", "C"]:
        return epc_threshold_scores.get(tech_type)
    else:
        return None


def score_dict_property_not_flat(tech_type, property_type):
    """
    Is the property NOT a flat?

    Is this property part of a building with multiple other properties (e.g. a flat)?
    """
    if property_type != "Flat, maisonette of apartment":
        return not_flat_scores.get(tech_type, tech_type)
    else:
        # Is this property part of a building with multiple other properties (e.g. a flat)?
        return multiple_props_scores.get(tech_type)


def score_dict_hot_water_tank_space(tech_type, water_tank_space):
    """
    Is there space for a hot water tank?
    """
    if water_tank_space:
        return water_tank_space_scores.get(tech_type)
    else:
        return None


def score_dict_garden_size(tech_type, outdoor_space):
    """
    Is the garden >10-25m2? (can change this number) * more for GSHP
    """
    if outdoor_space > 10:
        return garden_size_scores.get(tech_type)
    else:
        return None


def score_dict_external_space(tech_type, outdoor_space):
    """
    Is there some external space? (needs to be defined) *more for GSHP
    """
    if outdoor_space > 10:
        return external_space_scores.get(tech_type)
    else:
        return None


def score_dict_off_gas(
    tech_type,
    off_gas,
    # solar
):
    """
    Is it off-gas or does it have solar panels?
    """
    # TODO: add solar
    if off_gas:
        return offgas_scores.get(tech_type)
    else:
        return None


def score_dict_property_density(tech_type, lsoa):
    """
    Is there a high property density in this LSOA? (% TBC)
    """
    property_density_per_lsoa = {22: 0.8, 55: 0.2}  # To define
    threshold = 0.4
    if property_density_per_lsoa.get(lsoa) > threshold:
        return property_density_scores.get(tech_type)
    else:
        return None


def score_dict_high_heat_demand(tech_type, ruc_two_fold):
    """
    Is this property in an urban LSOA/high heat demand density LSOA?
    """
    if ruc_two_fold == "Urban":
        return high_heat_demand_scores.get(tech_type)
    else:
        return None


def score_dict_anchor_properties(tech_type, lsoa):
    """
    Is this property in a LSOA with schools/hospitals/community centres/housing developments?
    """
    anchors_per_lsoa = {22: True, 55: False}  # To define
    if anchors_per_lsoa.get(lsoa):
        return anchor_properties_scores.get(tech_type)
    else:
        return None


def get_property_scores(
    ruc_two_fold: str,
    off_gas: bool,
    property_density: float,
    outdoor_space: float,
    listed_building: bool,
    building_conservation_area: bool,
    property_type: str,
    epc_rating: str,
) -> dict:
    """
    Calculate heat pump suitability score per tech type for the standard (S) and Nesta (N) views for an EPC property.
    Tech types include: air-source heat pump (ASHP), ground-source heat pump (GSHP), shared ground loop (SGL), heat
    network (HN).

    Args:
        ruc_two_fold (str):
        off_gas (bool):
        property_density (float):
        outdoor_space (float):
        listed_building (bool):
        building_conservation_area (bool):
        property_type (str):
        epc_rating (str):

    Returns:
        dict: heat pump suitability scores per tech type.
    """
    all_scores = defaultdict(int)
    for scores in [
        score_dict_high_heat_demand(ruc_two_fold),
        score_dict_off_gas(off_gas),
        score_dict_property_density(property_density),
        score_dict_garden_size(outdoor_space),
        score_dict_external_space(outdoor_space),
        score_dict_site_regs(listed_building, building_conservation_area),
        score_dict_property_not_flat(property_type),
        score_dict_epc_rating(epc_rating),
    ]:
        if scores:
            for k, v in scores.items():
                all_scores[k] += v

    if not all_scores:
        all_scores = {
            "ASHP_S": 0,
            "ASHP_N": 0,
            "GSHP_S": 0,
            "GSHP_N": 0,
            "SGL_S": 0,
            "SGL_N": 0,
            "HN_S": 0,
            "HN_N": 0,
        }
    return all_scores


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
    density_threshold: int = 100,
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
        pl.when(pl.col("msoa_avg_outdoor_space_m2") > garden_threshold)
        .then(garden_size_scores.get(tech_type))
        .alias("garden_size_score"),
        pl.when(pl.col("msoa_avg_outdoor_space_m2") > external_space_threshold)
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
        pl.when(pl.col("msoa_avg_outdoor_space_m2").is_not_null())
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
            "msoa_avg_outdoor_space_m2",
            "listed_building_grade",
            "in_conservation_area",
            "property_type",
            "CURRENT_ENERGY_RATING",
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

    return scores_dict


if __name__ == "__main__":
    # TODO: logging.info not displaying to terminal for me
    logging.info("Loading EPC data with features")
    epc_df = get_enhanced_epc()

    logging.info("Filtering EPC data to rows with n_features >= minimum threshold")
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
        logging.info(f"Calculating suitability scores for tech type: {tech_type}")
        epc_scores_df = compute_df_avg_score_per_epc(epc_df, tech_type)
        scores.append(epc_scores_df)

    logging.info("Joining all scores to EPC dataset")
    for score_df in scores:
        epc_df = epc_df.join(score_df, on="UPRN", how="left")

    fs = s3fs.S3FileSystem()
    save_as = f"s3://asf-heat-pump-suitability/outputs/{datetime.today().strftime('%Y%m%d')}_2023_Q4_heat_pump_suitability_per_property.parquet"
    with fs.open(save_as, mode="wb") as f:
        epc_df.write_parquet(f)

    logging.info("Weighting scores and aggregating per LSOA")
    weighted_scores = []
    for lsoa_code in tqdm(epc_df["lsoa"].unique()):
        lsoa_df = epc_df.filter(pl.col("lsoa") == lsoa_code)
        lsoa_df = compute_df_weighted_score(lsoa_df)
        weighted_scores.append(compute_dict_lsoa_suitability_scores(lsoa_df, lsoa_code))

    logging.info("Saving LSOA heat pump suitability scores")
    suitability_df = pl.DataFrame(weighted_scores)
    fs = s3fs.S3FileSystem()
    save_as = f"s3://asf-heat-pump-suitability/outputs/{datetime.today().strftime('%Y%m%d')}_2023_Q4_heat_pump_suitability_per_lsoa.parquet"
    with fs.open(save_as, mode="wb") as f:
        suitability_df.write_parquet(f)
