"""
This can be run as a standalone script to calculate grid capacity per LSOA/DataZone in England, Scotland, and Wales.
Outputs will be saved to `outputs/reports/grid_capacity.csv` unless otherwise specified.
"""

import argparse
import logging
import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl

from asf_heat_pump_suitability.getters.get_dno_datasets import (
    generate_enw_gdf,
    generate_npg_gdf,
    generate_spen_gdf,
    generate_ssen_gdf,
    generate_ukpn_gdf,
    generate_wpd_gdf,
)
from asf_heat_pump_suitability.pipeline.prepare_features import (
    boundaries,
    household_count,
)

logger = logging.getLogger(__name__)

# Constants
CRS = "EPSG:4326"  # Geometry coordinate reference system - standard longitude/latitude projection
POWER_PER_HEATPUMP = 8  # Power rating per heat pump in kW
COEFFICIENT_OF_PERFORMANCE = 2.5  # Heat output (heat pump rating) / electricity consumed
SUBSTATION_SCALING_FACTOR = 1.0  # Factor to scale substation power rating (MVA) by


def _parse_capacity(value: Any) -> float:  # noqa: ANN401
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
    Generate a combined GeoDataFrame containing all primary substations and their service areas
    across Distribution Network Operators (DNOs).

    Combines data from:
        - Electricity North West (ENW)
        - Northern Powergrid (NPg)
        - Scottish Power Energy Networks (SPEN)
        - Scottish and Southern Electricity Networks (SSEN)
        - UK Power Networks (UKPN)
        - Western Power Distribution (WPD)

    Returns:
        gpd.GeoDataFrame: DataFrame with EPSG:4326 (WGS84) projection containing:
            - id: Unique substation identifier (format varies by operator)
            - firm_capacity_mva: Maximum power the substation can safely deliver (MVA)
            - peak_demand_mva: Maximum observed power demand at the substation (MVA)
            - geo_shape: Polygon geometry representing the substation's service area
            - operator: Distribution network operator code
                       (one of: "ENW", "NPg", "SPEN", "SSEN", "UKPN", "WPD")
    """

    substations_gdf = pd.concat(
        [
            generate_enw_gdf(),
            generate_npg_gdf(),
            generate_spen_gdf(),
            generate_ssen_gdf(),
            generate_ukpn_gdf(),
            generate_wpd_gdf(),
        ]
    )

    expected_operators = {"ENW", "NPg", "SPEN", "SSEN", "UKPN", "WPD"}
    actual_operators = set(substations_gdf["operator"])
    if actual_operators != expected_operators:
        missing = expected_operators - actual_operators
        extra = actual_operators - expected_operators
        raise ValueError(f"Unexpected operator set in substations data. Missing: {missing}, unexpected: {extra}")

    return substations_gdf


def distribute_substation_headroom(
    substations_gdf: gpd.GeoDataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    households_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribute substation headroom to LSOAs/DataZone, weighted by household count.

    Args:
        substations_gdf: GeoDataFrame of substations with headroom data.
        lsoa_gdf: GeoDataFrame of LSOA/DataZone boundaries.
        households_df: DataFrame with household counts per LSOA/DataZone.

    Returns:
        DataFrame with distributed headroom per LSOA/DataZone.
    """
    # Merge LSOA boundaries with household counts
    lsoa_with_households = lsoa_gdf.merge(households_df, on="lsoa", how="left")

    # Perform spatial join between LSOAs and substations
    joined = gpd.sjoin(lsoa_with_households, substations_gdf, how="inner", predicate="intersects").rename(
        columns={"id": "substation_id"}
    )

    # Calculate total households served by each substation
    substation_total_households = joined.groupby("substation_id")["households_count"].sum().reset_index()
    substation_total_households = substation_total_households.rename(columns={"households_count": "total_households"})

    # Merge total households back to joined dataframe
    joined = joined.merge(substation_total_households, on="substation_id")

    # Calculate the fraction of substation's load for each LSOA/DataZone
    joined["load_fraction"] = joined["households_count"] / joined["total_households"]

    # Distribute headroom based on load fraction
    joined["distributed_headroom"] = joined["headroom_mva"] * joined["load_fraction"]

    # Aggregate distributed headroom by LSOA/DataZone
    lsoa_headroom = joined.groupby("lsoa")["distributed_headroom"].sum().reset_index()

    return lsoa_headroom


def calculate_headroom_per_household(lsoa_headroom: pd.DataFrame, households_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate headroom per household for each LSOA/DataZone.

    Args:
        lsoa_headroom: DataFrame with LSOA/DataZone codes and distributed headroom.
        households_df: DataFrame with LSOA/DataZone codes and household counts.

    Returns:
        DataFrame with LSOA/DataZone codes, distributed headroom, household counts, and headroom per household.
    """
    merged = pd.merge(lsoa_headroom, households_df, on="lsoa", how="left")
    merged["headroom_per_household"] = merged["distributed_headroom"] / merged["households_count"]
    return merged[
        [
            "lsoa",
            "distributed_headroom",
            "households_count",
            "headroom_per_household",
        ]
    ]


def process_substation_lsoa_data(
    substations_gdf: gpd.GeoDataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    households_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process substation and LSOA/DataZone data to calculate distributed headroom per household.

    Args:
        substations_gdf: GeoDataFrame of substations with headroom data.
        lsoa_gdf: GeoDataFrame of LSOA/DataZone boundaries.
        households_df: DataFrame with household counts per LSOA/DataZone.

    Returns:
        DataFrame with processed data including distributed headroom per household.
    """
    lsoa_headroom = distribute_substation_headroom(substations_gdf, lsoa_gdf, households_df)
    return calculate_headroom_per_household(lsoa_headroom, households_df)


def assess_heatpump_suitability(
    lsoa_data: pd.DataFrame, power_per_heatpump: float, voltage_factor: float = 0.95
) -> pd.DataFrame:
    """
    Assess the heat pump installation capacity for LSOAs/DataZones based on grid capacity.

    Args:
        lsoa_data: DataFrame with LSOA/DataZone data including headroom and household counts.
        power_per_heatpump: Power consumption per heat pump in kW.
        voltage_factor: Factor to account for voltage drop (default 0.95).

    Returns:
        DataFrame with heat pump installation capacity assessment for each LSOA/DataZone.
    """
    # Convert headroom from MVA to kW
    lsoa_data["headroom_kw"] = lsoa_data["distributed_headroom"] * 1000 * voltage_factor

    # Calculate maximum number of heat pumps that could be installed
    lsoa_data["max_heatpumps"] = np.floor(lsoa_data["headroom_kw"] / power_per_heatpump)

    # Calculate percentage of households that could install heat pumps
    lsoa_data["heatpump_installation_percentage"] = (
        lsoa_data["max_heatpumps"] / lsoa_data["households_count"] * 100
    ).clip(upper=100, lower=0)

    # Calculate excess or deficit capacity
    lsoa_data["capacity_difference_kw"] = lsoa_data["headroom_kw"] - (
        lsoa_data["households_count"] * power_per_heatpump
    )

    return lsoa_data


def calculate_grid_capacity() -> pl.DataFrame:
    """
    Calculate the grid capacity for heat pump installations across all LSOAs / DataZones.

    This function performs the following steps:
    1. Generate and process substation data from all grid operators
    2. Load and process LSOA/DataZone and household data
    3. Distribute substation headroom to LSOAs
    4. Assess heat pump suitability based on available capacity

    Returns:
        DataFrame with grid capacity assessment results for each LSOA/DataZone
    """
    # Generate and process substation data
    substations = generate_substations_gdf().drop_duplicates(subset="id")
    substations["firm_capacity_mva"] = substations["firm_capacity_mva"].apply(_parse_capacity)
    substations["peak_demand_mva"] = substations["peak_demand_mva"].apply(_parse_capacity)

    # Apply substation rating scaling factor
    substations["firm_capacity_mva"] = substations["firm_capacity_mva"] * SUBSTATION_SCALING_FACTOR

    # Calculate headroom per substation
    substations["headroom_mva"] = substations["firm_capacity_mva"] - substations["peak_demand_mva"]

    # Load and process LSOA/DataZone boundary data
    lsoa_gdf = boundaries.load_transform_gdf_lsoa_dz_boundaries().to_crs(CRS)

    # Load and process household data
    households_df = household_count.load_transform_df_n_households().to_pandas()

    # Process substation and LSOA/DataZone data
    headroom_df = process_substation_lsoa_data(substations, lsoa_gdf, households_df)

    # Assess heat pump suitability
    consumption_per_heatpump = POWER_PER_HEATPUMP / COEFFICIENT_OF_PERFORMANCE
    result = assess_heatpump_suitability(headroom_df, consumption_per_heatpump)

    # Rename LSOA column for consistency

    return pl.from_pandas(result)


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
        default="outputs/reports/grid_capacity.csv",
    )

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = parse_arguments()

    grid_capacity_results = calculate_grid_capacity()
    logger.info("Grid capacity results (head):\n%s", grid_capacity_results.head())
    logger.info("Total LSOAs/DataZones assessed: %d", len(grid_capacity_results))
    logger.info(
        "Average LSOA/DataZone installation percentage: %s",
        grid_capacity_results["heatpump_installation_percentage"].mean(),
    )

    if args.save_as:
        grid_capacity_results.write_csv(args.save_as)
