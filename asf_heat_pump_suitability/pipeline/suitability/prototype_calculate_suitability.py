"""
Functions to calculate suitability of different HP technologies.
"""

import pandas as pd
import polars as pl
from collections import defaultdict
import logging
from typing import Optional

site_regs_scores = {
    "ASHP_S": 0.5,
    "ASHP_N": 0.5,
    "GSHP_S": 0.5,
    "GSHP_N": 0.5,
    "SGL_S": 0.5,
    "SGL_N": 0.5,
    "HN_S": 0.5,
    "HN_N": 0.5,
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
        "OFF_GAS",
        "property_density",
        "mso_avg_garden_size",
        "listed_building",
        "in_conservation_area",
        "PROPERTY_TYPE",
        "CURRENT_ENERGY_RATING",
    ]
    df = pl.read_parquet(
        "s3://asf-heat-pump-suitability/outputs/20240827_2023_Q4_EPC_weighted_features.parquet",
        columns=usecols,
    )
    return df


def score_dict_site_regs(conservation_zone, listed_building) -> Optional[dict]:
    """
    Is this property NOT listed / in a conservation zone / any other planning regulations?
    """
    if conservation_zone or listed_building:
        return None
    else:
        return site_regs_scores


def score_dict_grid_capacity(lsoa):
    """
    Does the electricity grid for the LSOA this property is in have capacity?
    """
    capacity_per_lsoa = {22: True, 55: False}  # To define

    if capacity_per_lsoa.get(lsoa):
        return grid_capacity_scores
    else:
        return None


def score_dict_epc_rating(epc):
    """
    Is the EPC rating >= C?
    """

    if epc in ["A", "B", "C"]:
        return epc_threshold_scores
    else:
        return None


def score_dict_property_not_flat(property_type):
    """
    Is the property NOT a flat?

    Is this property part of a building with multiple other properties (e.g. a flat)?
    """
    if property_type != "Flat, maisonette of apartment":
        return not_flat_scores
    else:
        # Is this property part of a building with multiple other properties (e.g. a flat)?
        return multiple_props_scores


def score_dict_hot_water_tank_space(water_tank_space):
    """
    Is there space for a hot water tank?
    """
    if water_tank_space:
        return water_tank_space_scores
    else:
        return None


def score_dict_garden_size(outdoor_space):
    """
    Is the garden >10-25m2? (can change this number) * more for GSHP
    """
    if outdoor_space > 10:
        return garden_size_scores
    else:
        return None


def score_dict_external_space(outdoor_space):
    """
    Is there some external space? (needs to be defined) *more for GSHP
    """
    if outdoor_space > 10:
        return external_space_scores
    else:
        return None


def score_dict_off_gas(
    off_gas,
    # solar
):
    """
    Is it off-gas or does it have solar panels?
    """
    # TODO: add solar
    if off_gas:
        return offgas_scores
    else:
        return None


def score_dict_property_density(lsoa):
    """
    Is there a high property density in this LSOA? (% TBC)
    """
    property_density_per_lsoa = {22: 0.8, 55: 0.2}  # To define
    threshold = 0.4
    if property_density_per_lsoa.get(lsoa) > threshold:
        return property_density_scores
    else:
        return None


def score_dict_high_heat_demand(ruc_two_fold):
    """
    Is this property in an urban LSOA/high heat demand density LSOA?
    """
    if ruc_two_fold == "Urban":
        return high_heat_demand_scores
    else:
        return None


def score_dict_anchor_properties(lsoa):
    """
    Is this property in a LSOA with schools/hospitals/community centres/housing developments?
    """
    anchors_per_lsoa = {22: True, 55: False}  # To define
    if anchors_per_lsoa.get(lsoa):
        return anchor_properties_scores
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
