"""
Functions to calculate suitability of different HP technologies.
"""

import pandas as pd
import polars as pl
from collections import defaultdict
import logging
from typing import Optional

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
    return df


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


if __name__ == "__main__":
    print("Loading EPC data with features")
    epc_enhanced_data = get_enhanced_epc()

    # For now, just include some of the measurements in the calculation - later add the rest.
    print("Calculating max scores")
    max_scores = defaultdict(int)
    for scores in [
        garden_size_scores,
        external_space_scores,
        not_flat_scores,
        high_heat_demand_scores,
        not_flat_scores,
        multiple_props_scores,
        epc_threshold_scores,
    ]:
        for k, v in scores.items():
            max_scores[k] += v

    print("Calculating suitability score of each tech per EPC record")
    tech_suitability = epc_enhanced_data.apply(
        lambda x: {
            k: v / max_scores.get(k)
            for k, v in get_property_scores(
                x["ruc_two_fold"],  # Urban rural
                # Off gas
                # Property density
                x["msoa_avg_outdoor_space_m2"],  # Avg garden space
                # Estimated garden space
                # Listed building status
                # Building conservation area
                x["PROPERTY_TYPE"],  # Property type (flats)
                x["CURRENT_ENERGY_RATING"],  # EPC rating
            ).items()
        },
        axis=1,
    ).apply(pd.Series)

    print("Joining tech suitability score to EPC dataset")
    epc_enhanced_data = epc_enhanced_data.join(tech_suitability)

    print("Calculating mean score per lsoa")
    average_suitability_per_lsoa = epc_enhanced_data.groupby("lsoa")[
        tech_suitability.columns
    ].mean()
