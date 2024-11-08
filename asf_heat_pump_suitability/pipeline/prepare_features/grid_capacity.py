import re
import os
from typing import Any
import logging
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd

from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.getters.get_dno_datasets import (
    generate_enw_gdf,
    generate_npg_gdf,
    generate_spen_gdf,
    generate_ssen_gdf,
    generate_ukpn_gdf,
    generate_wpd_gdf,
)

# Constants
CRS = "EPSG:4326"  # Geometry coordinate reference system - standard longitude/latitude projection
POWER_PER_HEATPUMP = 8  # Power rating per heat pump in kW
COEFFICIENT_OF_PERFORMANCE = (
    2.5  # Heat output (heat pump rating) / electricity consumed
)
SUBSTATION_SCALING_FACTOR = 1.0  # Factor to scale substation power rating (MVA) by


def _parse_capacity(value: Any) -> float:
    """
    Parse complex capacity strings and return total capacity as float.

    Args:
        value: Input capacity value in various formats.

    Returns:
        Parsed capacity as a float, or NaN if parsing fails.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip().lower()

    # Pattern to match "X x Y" format (e.g., "2 x 12.5")
    pattern = r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)"
    match = re.match(pattern, value)

    if match:
        count = float(match.group(1))
        individual_capacity = float(match.group(2))
        return count * individual_capacity

    try:
        return float(value)
    except ValueError:
        logging.warning(f"Could not parse capacity value: {value}")
        return np.nan


def generate_substations_gdf() -> gpd.GeoDataFrame:
    """
    Generate a combined GeoDataFrame for all grid operators' substations.

    Returns:
        GeoDataFrame with combined substation data from all operators.
    """
    return pd.concat(
        [
            generate_enw_gdf(),
            generate_npg_gdf(),
            generate_spen_gdf(),
            generate_ssen_gdf(),
            generate_ukpn_gdf(),
            generate_wpd_gdf(),
        ]
    )


def distribute_substation_headroom(
    substations_gdf: gpd.GeoDataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    households_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribute substation headroom to LSOAs, weighted by household count.

    Args:
        substations_gdf: GeoDataFrame of substations with headroom data.
        lsoa_gdf: GeoDataFrame of LSOA boundaries.
        households_df: DataFrame with household counts per LSOA.

    Returns:
        DataFrame with distributed headroom per LSOA.
    """
    # Merge LSOA boundaries with household counts
    lsoa_with_households = lsoa_gdf.merge(households_df, on="LSOA21CD", how="left")

    # Perform spatial join between LSOAs and substations
    joined = gpd.sjoin(
        lsoa_with_households, substations_gdf, how="inner", predicate="intersects"
    ).rename(columns={"index_right": "substation_id"})

    # Calculate total households served by each substation
    substation_total_households = (
        joined.groupby("substation_id")["household_count"].sum().reset_index()
    )
    substation_total_households = substation_total_households.rename(
        columns={"household_count": "total_households"}
    )

    # Merge total households back to joined dataframe
    joined = joined.merge(substation_total_households, on="substation_id")

    # Calculate the fraction of substation's load for each LSOA
    joined["load_fraction"] = joined["household_count"] / joined["total_households"]

    # Distribute headroom based on load fraction
    joined["distributed_headroom"] = joined["headroom_mva"] * joined["load_fraction"]

    # Aggregate distributed headroom by LSOA
    lsoa_headroom = (
        joined.groupby("LSOA21CD")["distributed_headroom"].sum().reset_index()
    )

    return lsoa_headroom


def calculate_headroom_per_household(
    lsoa_headroom: pd.DataFrame, households_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate headroom per household for each LSOA.

    Args:
        lsoa_headroom: DataFrame with LSOA codes and distributed headroom.
        households_df: DataFrame with LSOA codes and household counts.

    Returns:
        DataFrame with LSOA codes, distributed headroom, household counts, and headroom per household.
    """
    merged = pd.merge(lsoa_headroom, households_df, on="LSOA21CD", how="left")
    merged["headroom_per_household"] = (
        merged["distributed_headroom"] / merged["household_count"]
    )
    return merged[
        [
            "LSOA21CD",
            "distributed_headroom",
            "household_count",
            "headroom_per_household",
        ]
    ]


def process_substation_lsoa_data(
    substations_gdf: gpd.GeoDataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    households_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process substation and LSOA data to calculate distributed headroom per household.

    Args:
        substations_gdf: GeoDataFrame of substations with headroom data.
        lsoa_gdf: GeoDataFrame of LSOA boundaries.
        households_df: DataFrame with household counts per LSOA.

    Returns:
        DataFrame with processed data including distributed headroom per household.
    """
    lsoa_headroom = distribute_substation_headroom(
        substations_gdf, lsoa_gdf, households_df
    )
    return calculate_headroom_per_household(lsoa_headroom, households_df)


def assess_heatpump_suitability(
    lsoa_data: pd.DataFrame, power_per_heatpump: float, voltage_factor: float = 0.95
) -> pd.DataFrame:
    """
    Assess the heat pump installation capacity for LSOAs based on grid capacity.

    Args:
        lsoa_data: DataFrame with LSOA data including headroom and household counts.
        power_per_heatpump: Power consumption per heat pump in kW.
        voltage_factor: Factor to account for voltage drop (default 0.95).

    Returns:
        DataFrame with heat pump installation capacity assessment for each LSOA.
    """
    # Convert headroom from MVA to kW
    lsoa_data["headroom_kw"] = lsoa_data["distributed_headroom"] * 1000 * voltage_factor

    # Calculate maximum number of heat pumps that could be installed
    lsoa_data["max_heatpumps"] = np.floor(lsoa_data["headroom_kw"] / power_per_heatpump)

    # Calculate percentage of households that could install heat pumps
    lsoa_data["heatpump_installation_percentage"] = (
        lsoa_data["max_heatpumps"] / lsoa_data["household_count"] * 100
    ).clip(upper=100, lower=0)

    # Calculate excess or deficit capacity
    lsoa_data["capacity_difference_kw"] = lsoa_data["headroom_kw"] - (
        lsoa_data["household_count"] * power_per_heatpump
    )

    return lsoa_data


def calculate_grid_capacity() -> pd.DataFrame:
    """
    Calculate the grid capacity for heat pump installations across all LSOAs.

    This function performs the following steps:
    1. Generate and process substation data from all grid operators
    2. Load and process LSOA and household data
    3. Distribute substation headroom to LSOAs
    4. Assess heat pump suitability based on available capacity

    Returns:
        DataFrame with grid capacity assessment results for each LSOA
    """
    # Generate and process substation data
    substations = generate_substations_gdf().drop_duplicates(subset="id")
    substations["firm_capacity_mva"] = substations["firm_capacity_mva"].apply(
        _parse_capacity
    )
    substations["peak_demand_mva"] = substations["peak_demand_mva"].apply(
        _parse_capacity
    )

    # Apply substation rating scaling factor
    substations["firm_capacity_mva"] = (
        substations["firm_capacity_mva"] * SUBSTATION_SCALING_FACTOR
    )

    # Calculate headroom per substation
    substations["headroom_mva"] = (
        substations["firm_capacity_mva"] - substations["peak_demand_mva"]
    )

    # Load and process LSOA boundary data
    lsoa_gdf = gpd.read_file(
        "s3://asf-heat-pump-suitability/source_data/Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/LSOA_2021_EW_BFE_V9.shp"
    ).to_crs(CRS)

    # Load and process household data
    households_df = get_datasets.get_df_ons_number_of_households()
    households_df = households_df.rename(
        mapping={"mnemonic": "LSOA21CD", "2021": "household_count"}
    ).to_pandas()

    # Process substation and LSOA data
    headroom_df = process_substation_lsoa_data(substations, lsoa_gdf, households_df)

    # Assess heat pump suitability
    consumption_per_heatpump = POWER_PER_HEATPUMP / COEFFICIENT_OF_PERFORMANCE
    result = assess_heatpump_suitability(headroom_df, consumption_per_heatpump)

    # Rename LSOA column for consistency
    result = result.rename(columns={"LSOA21CD": "lsoa"})

    return result


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_as",
        help="Path to save grid capacity results to. If unspecified, save with default filename.",
        type=str,
        required=False,
        default="grid_capacity.csv",
    )

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = parse_arguments()

    grid_capacity_results = calculate_grid_capacity()
    print(grid_capacity_results.head())
    print(f"Total LSOAs assessed: {len(grid_capacity_results)}")
    print(
        "Average LSOA installation percentage: "
        + str(grid_capacity_results["heatpump_installation_percentage"].mean())
    )

    if args.save_as:
        grid_capacity_results.to_csv(args.save_as, index=False)
