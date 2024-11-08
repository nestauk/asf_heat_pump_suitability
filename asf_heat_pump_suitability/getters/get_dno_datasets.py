import re
from typing import Any
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

from asf_heat_pump_suitability.utils.geo_utils import parse_binary_geometry
from asf_heat_pump_suitability.getters.s3_getters import (
    load_s3_data,
)

CRS = "EPSG:4326"  # Geometry coordinate reference system - standard longitude/latitude projection


def generate_enw_gdf() -> gpd.GeoDataFrame:
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
    enw["geopoint"] = enw["geopoint"].apply(parse_binary_geometry)

    # Create GeoDataFrame and perform spatial join with shape data
    enw_gdf = gpd.GeoDataFrame(enw, geometry="geopoint", crs=CRS)
    enw_shape["geo_shape"] = enw_shape["geo_shape"].apply(parse_binary_geometry)
    enw_shape["geo_point_2d"] = enw_shape["geo_point_2d"].apply(parse_binary_geometry)
    enw_shape_gdf = gpd.GeoDataFrame(enw_shape, geometry="geo_shape", crs=CRS)
    enw_df = gpd.sjoin(enw_gdf, enw_shape_gdf, how="right", predicate="intersects")
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


def generate_npg_gdf() -> gpd.GeoDataFrame:
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
    npg["location"] = npg["location"].apply(parse_binary_geometry)

    # Filter demand data
    npg_demand = npg_demand[
        (npg_demand["bulk_supply_point_or_primary"] == "Primary")
        & (npg_demand["scenario_name"] == "NPg Best View")
    ]
    npg_demand["geo_point_2d"] = npg_demand["geo_point_2d"].apply(parse_binary_geometry)
    npg_demand["geo_shape"] = npg_demand["geo_shape"].apply(parse_binary_geometry)
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


def generate_spen_gdf() -> gpd.GeoDataFrame:
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
    spen_df["geo_point_2d"] = spen_df["geo_point_2d"].apply(parse_binary_geometry)
    spen_df["geo_shape"] = spen_df["geo_shape"].apply(parse_binary_geometry)

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


def generate_ssen_gdf() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for Scottish and Southern Electricity Networks (SSEN) substations.

    Returns:
        GeoDataFrame with SSEN substation data.
    """
    ssen_demand = load_s3_data(
        "asf-heat-pump-suitability",
        "source_data/grid_operators/SSEN/demand-heat-map-update-feb-24.xlsx - PS.csv",
    )
    sepd_shape = gpd.read_file(
        "s3://asf-heat-pump-suitability/source_data/grid_operators/SSEN/SEPD Primary Sub Boundaries"
    )
    shepd_shape = gpd.read_file(
        "s3://asf-heat-pump-suitability/source_data/grid_operators/SSEN/SHEPD Primary Sub Boundaries"
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

    ssen_df["operator"] = "SSEN"

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


def generate_ukpn_gdf() -> gpd.GeoDataFrame:
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

    # Calculate the headroom
    # The "demand" column is the percentage of total capacity that the substation has available at peak load
    # So to calculate the actual value in MVA, we need to multiply demand headroom by capacity
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
    ukpn["geo_point_2d"] = ukpn["geo_point_2d"].apply(parse_binary_geometry)
    ukpn["geo_shape"] = ukpn["geo_shape"].apply(parse_binary_geometry)

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


def generate_wpd_gdf() -> gpd.GeoDataFrame:
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
