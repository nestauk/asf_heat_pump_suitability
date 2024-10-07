import re
import os
import shutil
from typing import Union, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely import wkb

from asf_heat_pump_suitability.getters.s3_getters import (
    load_s3_data,
    get_shapefile_from_s3,
)
from asf_heat_pump_suitability.getters import get_datasets

# Constants
CRS = "EPSG:4326"  # Geometry coordinate reference system - standard longitude/latitude projection
TAKEUP_RATE = 30  # Percentage of households in each LSOA to install heat pumps
POWER_PER_HEATPUMP = 8  # Power rating per heat pump in kW
COEFFICIENT_OF_PERFORMANCE = (
    2.5  # Heat output (heat pump rating) / electricity consumed
)
SUBSTATION_SCALING_FACTOR = 1.0  # Factor to scale substation power rating (MVA) by


def _parse_binary_geometry(
    binary_data: Union[BaseGeometry, bytes, str]
) -> Union[BaseGeometry, None]:
    """
    Parse binary geometry data into Shapely geometry object.

    Args:
        binary_data: Input geometry data in various formats.

    Returns:
        Shapely geometry object or None if parsing fails.
    """
    if isinstance(binary_data, BaseGeometry):
        return binary_data
    elif isinstance(binary_data, bytes):
        return wkb.loads(binary_data)
    elif isinstance(binary_data, str):
        return wkb.loads(binary_data, hex=True)
    else:
        return None


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
        print(f"Warning: Could not parse capacity value: {value}")
        return np.nan


def process_shapefile(shapefile_directory: str) -> gpd.GeoDataFrame:
    """
    Process a shapefile from S3 and return it as a GeoDataFrame.

    Args:
        shapefile_directory: S3 directory containing the shapefile.

    Returns:
        GeoDataFrame containing the processed shapefile data.
    """
    bucket_name = "asf-heat-pump-suitability"
    temp_dir = get_shapefile_from_s3(bucket_name, shapefile_directory)

    try:
        shp_file = next(f for f in os.listdir(temp_dir) if f.endswith(".shp"))
        shp_path = os.path.join(temp_dir, shp_file)
        gdf = gpd.read_file(shp_path)
        gdf = gdf.to_crs(epsg=27700)  # Convert to British National Grid
        return gdf
    finally:
        shutil.rmtree(temp_dir)


def generate_enw_df() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for Electricity North West (ENW) substations.

    Returns:
        GeoDataFrame with ENW substation data.
    """
    enw_demand = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/ENW/dfes-2023-primary-data0.parquet",
    )
    enw_substations = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/ENW/ndp-pry-bsp-headroom.parquet",
    )
    enw_shape = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/ENW/ndp-pry-voronoi.parquet",
    )

    # Filter data for primary substations, current year, and best view scenario
    enw_substations = enw_substations[
        (enw_substations["substation_type"] == "PRIMARY")
        & (enw_substations["year"] == "2024")
        & (enw_substations["scenario"] == "1 - BEST VIEW")
        & (enw_substations["status"] == "FIRM")
    ]
    enw_demand = enw_demand[
        (enw_demand.group == "PRIMARY")
        & (enw_demand.year == "2024")
        & (enw_demand.scenario == "Best View")
    ]

    # Merge substation and demand data
    enw = pd.merge(
        enw_substations[["substation", "headroom_mva", "geopoint"]],
        enw_demand[["number", "maximum_demand_mva_per_primary_substation"]],
        left_on="substation",
        right_on="number",
    )

    # Calculate firm capacity and add operator information
    enw["firm_capacity_mva"] = (
        enw["maximum_demand_mva_per_primary_substation"] + enw["headroom_mva"]
    )
    enw["operator"] = "ENW"
    enw["geopoint"] = enw["geopoint"].apply(_parse_binary_geometry)

    # Create GeoDataFrame and perform spatial join with shape data
    enw_gdf = gpd.GeoDataFrame(enw, geometry="geopoint", crs=CRS)
    enw_shape["geo_shape"] = enw_shape["geo_shape"].apply(_parse_binary_geometry)
    enw_shape["geo_point_2d"] = enw_shape["geo_point_2d"].apply(_parse_binary_geometry)
    enw_shape_gdf = gpd.GeoDataFrame(enw_shape, geometry="geo_shape", crs=CRS)
    enw_df = gpd.sjoin(enw_gdf, enw_shape_gdf, how="right", predicate="within")
    enw_df = enw_df.dropna(subset=["index_left"])

    # Prepare final dataframe
    enw_df["substation_id"] = enw_df["pry_group"] + "_" + enw_df["substation"]
    return enw_df[
        [
            "substation_id",
            "firm_capacity_mva",
            "maximum_demand_mva_per_primary_substation",
            "geo_shape",
            "operator",
        ]
    ].rename(
        columns={
            "substation_id": "id",
            "maximum_demand_mva_per_primary_substation": "peak_demand_mva",
            "geo_shape": "geo_shape",
        }
    )


def generate_npg_df() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for Northern Powergrid (NPg) substations.

    Returns:
        GeoDataFrame with NPg substation data.
    """
    npg_substations = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/NPg/heatmapdemanddata.parquet",
    )
    npg_demand = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/NPg/npg_ndp_demand_headroom.parquet",
    )

    # Filter and process substation data
    npg_substations = npg_substations[npg_substations.substation_class == "Primary"]
    npg = npg_substations[
        [
            "substation_id",
            "substation_name",
            "firm_capacity_load_mva",
            "maximum_demand_mva",
            "location",
        ]
    ]
    npg["headroom_mva"] = (
        npg_substations["firm_capacity_load_mva"]
        - npg_substations["maximum_demand_mva"]
    )
    npg["operator"] = "NPg"
    npg["location"] = npg["location"].apply(_parse_binary_geometry)

    # Filter demand data
    npg_demand = npg_demand[
        (npg_demand["bulk_supply_point_or_primary"] == "Primary")
        & (npg_demand["scenario_name"] == "NPg Best View")
    ]
    npg_demand["geo_point_2d"] = npg_demand["geo_point_2d"].apply(
        _parse_binary_geometry
    )
    npg_demand["geo_shape"] = npg_demand["geo_shape"].apply(_parse_binary_geometry)
    npg_shape = npg_demand[["geo_point_2d", "geo_shape"]].drop_duplicates()

    # Create GeoDataFrames and perform spatial join
    npg_gdf = gpd.GeoDataFrame(npg, geometry="location", crs=CRS)
    npg_shape_gdf = gpd.GeoDataFrame(npg_shape, geometry="geo_shape", crs=CRS)
    npg_df = gpd.sjoin(npg_shape_gdf, npg_gdf, how="left", predicate="intersects")
    npg_df = npg_df.dropna(subset="index_right")

    # Prepare final dataframe
    npg_df["id"] = npg_df["substation_name"] + "_" + npg_df["substation_id"]
    return npg_df[
        ["id", "firm_capacity_load_mva", "maximum_demand_mva", "geo_shape", "operator"]
    ].rename(
        columns={
            "firm_capacity_load_mva": "firm_capacity_mva",
            "maximum_demand_mva": "peak_demand_mva",
        }
    )


def generate_spen_df() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for Scottish Power Energy Networks (SPEN) substations.

    Returns:
        GeoDataFrame with SPEN substation data.
    """
    spen_spd_substations = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/SPEN/distributed-generation-sp-distribution-heat-maps-spd-primary-substations.parquet",
    )
    spen_spm_substations = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/SPEN/distributed-generation-sp-manweb-heat-maps-spm-primary-substations.parquet",
    )
    spen_spd_shape = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/SPEN/ndp-spd-primary-substation-polygons.parquet",
    )
    spen_spm_shape = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/SPEN/ndp-spm-primary-group-polygons.parquet",
    )

    # Process SPM data
    spm_df = spen_spm_substations.merge(spen_spm_shape, how="left", on="primary_group")
    spm_df = spm_df[
        [
            "substation_name",
            "primary_group",
            "firm_capacity_mva",
            "maximum_load_mva",
            "geo_point_2d",
            "geo_shape",
        ]
    ]

    # Process SPD data
    spd_df = spen_spd_substations.merge(
        spen_spd_shape,
        how="left",
        left_on="substation_name",
        right_on="primary_substation",
    )
    spd_df = spd_df[
        [
            "substation_name",
            "firm_capacity_mva",
            "maximum_load_mva",
            "geo_point_2d",
            "geo_shape",
        ]
    ]

    # Combine SPM and SPD data
    spen_df = pd.concat([spd_df, spm_df])
    spen_df["geo_point_2d"] = spen_df["geo_point_2d"].apply(_parse_binary_geometry)
    spen_df["geo_shape"] = spen_df["geo_shape"].apply(_parse_binary_geometry)

    spen_df["operator"] = "SPEN"

    return gpd.GeoDataFrame(
        spen_df[
            [
                "substation_name",
                "firm_capacity_mva",
                "maximum_load_mva",
                "geo_shape",
                "operator",
            ]
        ].rename(
            columns={"substation_name": "id", "maximum_load_mva": "peak_demand_mva"}
        ),
        geometry="geo_shape",
        crs=CRS,
    )


def generate_ssen_df() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for Scottish and Southern Electricity Networks (SSEN) substations.

    Returns:
        GeoDataFrame with SSEN substation data.
    """
    ssen_demand = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/SSEN/demand-heat-map-update-feb-24.xlsx - PS.csv",
    )
    sepd_shape = process_shapefile(
        "source_data/grid_operators/SSEN/SEPD Primary Sub Boundaries"
    )
    shepd_shape = process_shapefile(
        "source_data/grid_operators/SSEN/SHEPD Primary Sub Boundaries"
    )

    # Process SSEN data
    ssen = ssen_demand[
        [
            "Primary Substation Name",
            "Nameplate rating (MVA)",
            "Maximum Load (MVA)",
            "Latitude",
            "Longitude",
            "Grid Reference",
        ]
    ]
    ssen_shape = pd.concat([shepd_shape, sepd_shape])
    ssen_df = ssen.merge(
        ssen_shape, left_on="Primary Substation Name", right_on="Primary"
    )

    ssen_df["operator"] = "UKPN"

    ssen_df = gpd.GeoDataFrame(
        ssen_df[
            [
                "Primary Substation Name",
                "Nameplate rating (MVA)",
                "Maximum Load (MVA)",
                "geometry",
                "operator",
            ]
        ].rename(
            columns={
                "Primary Substation Name": "id",
                "Nameplate rating (MVA)": "firm_capacity_mva",
                "Maximum Load (MVA)": "peak_demand_mva",
                "geometry": "geo_shape",
            }
        ),
        geometry="geo_shape",
        crs="EPSG:27700",
    ).to_crs(CRS)

    return ssen_df[ssen_df.geometry.is_valid]


def generate_ukpn_df() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for UK Power Networks (UKPN) substations.

    Returns:
        GeoDataFrame with UKPN substation data.
    """
    ukpn_primary_substations = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/UKPN/ukpn_primary_postcode_area.parquet",
    )

    # Process UKPN data
    ukpn_primary_substations["firm_capacity_mva"] = ukpn_primary_substations.apply(
        lambda x: (
            x["firmcapacitywinter"]
            if x["seasonofconstraint"] == "Winter"
            else x["firmcapacitysummer"]
        ),
        axis=1,
    )
    ukpn_primary_substations["headroom_mva"] = (
        ukpn_primary_substations["firm_capacity_mva"]
        * ukpn_primary_substations["demand"]
        / 100
    )
    ukpn_primary_substations["maximum_load_mva"] = (
        ukpn_primary_substations["firm_capacity_mva"]
        - ukpn_primary_substations["headroom_mva"]
    )

    ukpn = ukpn_primary_substations[
        [
            "primary",
            "primarysubstationname",
            "firm_capacity_mva",
            "maximum_load_mva",
            "headroom_mva",
            "geo_point_2d",
            "geo_shape",
        ]
    ]
    ukpn["geo_point_2d"] = ukpn["geo_point_2d"].apply(_parse_binary_geometry)
    ukpn["geo_shape"] = ukpn["geo_shape"].apply(_parse_binary_geometry)

    ukpn["operator"] = "UKPN"

    return gpd.GeoDataFrame(
        ukpn[
            [
                "primary",
                "firm_capacity_mva",
                "maximum_load_mva",
                "geo_shape",
                "operator",
            ]
        ].rename(columns={"primary": "id", "maximum_load_mva": "peak_demand_mva"}),
        geometry="geo_shape",
        crs=CRS,
    )


def generate_wpd_df() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for Western Power Distribution (WPD) substations.

    Returns:
        GeoDataFrame with WPD substation data.
    """
    wpd_substations = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/WPD/wpd-network-capacity-map.csv",
    )
    wpd_regions = [
        "east-midlands-primary.gpkg",
        "south-wales-primary.gpkg",
        "south-west-primary.gpkg",
        "west-midlands-primary.gpkg",
    ]
    wpd_shapes = []

    # Load and concatenate shape files for different regions
    for region in wpd_regions:
        wpd_shapes.append(
            load_s3_data(
                "asf-heat-pump-suitability", f"source_data/grid_operators/WPD/{region}"
            )
        )
    wpd_shapes = pd.concat(wpd_shapes)

    # Process WPD data
    wpd_substations = wpd_substations[wpd_substations["Asset_Type"] == "Primary"]
    wpd_substations["headroom_mva"] = (
        wpd_substations["Firm_Capacity_of_Substation_(MVA)"]
        - wpd_substations["Measured_Peak_Demand_(MVA)"]
    )
    wpd = wpd_substations[
        [
            "Substation_Name",
            "Firm_Capacity_of_Substation_(MVA)",
            "Measured_Peak_Demand_(MVA)",
            "headroom_mva",
            "Network_Reference_ID",
        ]
    ]

    # Merge with shape data
    wpd_df = wpd.merge(wpd_shapes, left_on="Network_Reference_ID", right_on="PRIM_NRID")

    wpd_df["operator"] = "WPD"

    return gpd.GeoDataFrame(
        wpd_df[
            [
                "Substation_Name",
                "Firm_Capacity_of_Substation_(MVA)",
                "Measured_Peak_Demand_(MVA)",
                "geometry",
                "operator",
            ]
        ].rename(
            columns={
                "Substation_Name": "id",
                "Firm_Capacity_of_Substation_(MVA)": "firm_capacity_mva",
                "Measured_Peak_Demand_(MVA)": "peak_demand_mva",
                "geometry": "geo_shape",
            }
        ),
        geometry="geo_shape",
        crs="EPSG:27700",
    ).to_crs(CRS)


def generate_substations_df() -> gpd.GeoDataFrame:
    """
    Generate a combined GeoDataFrame for all grid operators' substations.

    Returns:
        GeoDataFrame with combined substation data from all operators.
    """
    return pd.concat(
        [
            generate_enw_df(),
            generate_npg_df(),
            generate_spen_df(),
            generate_ssen_df(),
            generate_ukpn_df(),
            generate_wpd_df(),
        ]
    )


def aggregate_oa_to_lsoa(oa_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Aggregate Output Area (OA) geometries to Lower Layer Super Output Area (LSOA) level.

    Args:
        oa_gdf: GeoDataFrame containing OA-level data with LSOA codes.

    Returns:
        GeoDataFrame aggregated to LSOA level.
    """
    if oa_gdf.crs is None:
        raise ValueError(
            "Input GeoDataFrame must have a defined coordinate reference system (CRS)"
        )

    lsoa_gdf = oa_gdf.dissolve(by="LSOA21CD", as_index=False)
    return lsoa_gdf.reset_index(drop=True)


def distribute_substation_headroom(
    substations_df: gpd.GeoDataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    households_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribute substation headroom to LSOAs, weighted by household count.

    Args:
        substations_df: GeoDataFrame of substations with headroom data.
        lsoa_gdf: GeoDataFrame of LSOA boundaries.
        households_df: DataFrame with household counts per LSOA.

    Returns:
        DataFrame with distributed headroom per LSOA.
    """
    # Merge LSOA boundaries with household counts
    lsoa_with_households = lsoa_gdf.merge(households_df, on="LSOA21CD", how="left")

    # Perform spatial join between LSOAs and substations
    joined = gpd.sjoin(
        lsoa_with_households, substations_df, how="inner", predicate="intersects"
    )

    # Calculate total households served by each substation
    substation_total_households = (
        joined.groupby("index_right")["household_count"].sum().reset_index()
    )
    substation_total_households = substation_total_households.rename(
        columns={"household_count": "total_households"}
    )

    # Merge total households back to joined dataframe
    joined = joined.merge(substation_total_households, on="index_right")

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
    substations_df: gpd.GeoDataFrame,
    lsoa_gdf: gpd.GeoDataFrame,
    households_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process substation and LSOA data to calculate distributed headroom per household.

    Args:
        substations_df: GeoDataFrame of substations with headroom data.
        lsoa_gdf: GeoDataFrame of LSOA boundaries.
        households_df: DataFrame with household counts per LSOA.

    Returns:
        DataFrame with processed data including distributed headroom per household.
    """
    lsoa_headroom = distribute_substation_headroom(
        substations_df, lsoa_gdf, households_df
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
    substations = generate_substations_df()
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
    lsoa_boundaries = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/Output_Areas_2021_EW_BGC_V2_4299916833741807639.geojson",
    )
    lsoa_gdf = aggregate_oa_to_lsoa(lsoa_boundaries)

    # Load and process household data
    households_df = get_datasets.get_df_ons_number_of_households()
    households_df = households_df.rename(
        mapping={"mnemonic": "LSOA21CD", "2021": "household_count"}
    ).to_pandas()

    # Process substation and LSOA data
    headroom_df = process_substation_lsoa_data(substations, lsoa_gdf, households_df)

    # Assess heat pump suitability
    consumption_per_heatpump = POWER_PER_HEATPUMP / COEFFICIENT_OF_PERFORMANCE
    result = assess_heatpump_suitability(
        headroom_df, consumption_per_heatpump, TAKEUP_RATE
    )

    # Rename LSOA column for consistency
    result = result.rename(columns={"LSOA21CD": "lsoa21"})

    return result


# Example usage
if __name__ == "__main__":
    grid_capacity_results = calculate_grid_capacity()
    print(grid_capacity_results.head())
    print(f"Total LSOAs assessed: {len(grid_capacity_results)}")
    print(
        f"LSOAs with sufficient grid capacity: {grid_capacity_results['has_grid_capacity'].sum()}"
    )
    print(
        f"Percentage of LSOAs with sufficient capacity: {(grid_capacity_results['has_grid_capacity'].mean() * 100):.2f}%"
    )

    # Optionally, save results to a file
    # grid_capacity_results.to_csv("grid_capacity_assessment_results.csv", index=False)
