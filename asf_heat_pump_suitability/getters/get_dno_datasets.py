import pandas as pd
import geopandas as gpd
import polars as pl

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.utils.geo_utils import parse_binary_geometry

CRS = "EPSG:4326"  # Geometry coordinate reference system - standard longitude/latitude projection


def generate_enw_gdf() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for Electricity North West (ENW) primary substations and their service areas.

    The function processes substation data from ENW's Distribution Future Energy Scenarios (DFES)
    and Network Development Plan (NDP) datasets. It combines point locations of substations with
    their corresponding service area polygons (Voronoi polygons) and capacity/demand information.

    Returns:
        gpd.GeoDataFrame: DataFrame with EPSG:4326 (WGS84) projection containing:
            - id: Unique substation identifier
            - firm_capacity_mva: Maximum power the substation can safely deliver (MVA)
            - peak_demand_mva: Maximum observed power demand at the substation (MVA)
            - geo_shape: Polygon geometry representing the substation's service area
            - operator: Distribution network operator code ("ENW")
    """
    enw_demand = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["E_ENW_dfes_primaries"]
    ).to_pandas()
    enw_substations = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["E_ENW_ndp_headroom"]
    ).to_pandas()
    enw_shape = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["E_ENW_ndp_voronoi"]
    ).to_pandas()

    # Filter data for primary substations, current year, and best view scenario
    # Best view scenario represents middle ground future projections of demand on substation capacity
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
    enw["geopoint"] = enw["geopoint"].apply(parse_binary_geometry)

    # Distribution areas are given by primary group rather than individual primary substations
    # Substation data doesn't contain information on primary group, so we need to perform a spatial join
    enw_gdf = gpd.GeoDataFrame(enw, geometry="geopoint", crs=CRS)
    # polygon geometry of substation distribution area
    enw_shape["geo_shape"] = enw_shape["geo_shape"].apply(parse_binary_geometry)
    enw_shape_gdf = gpd.GeoDataFrame(enw_shape, geometry="geo_shape", crs=CRS)
    enw_df = gpd.sjoin(enw_gdf, enw_shape_gdf, how="right", predicate="within")
    enw_df = enw_df.dropna(subset=["index_left"])

    # Aggregate capacity and demand data to primary group level
    enw_df = (
        enw_df.groupby(["pry_group", "geo_shape"])[
            ["firm_capacity_mva", "maximum_demand_mva_per_primary_substation"]
        ]
        .sum()
        .reset_index()
    )
    enw_df["operator"] = "ENW"
    return gpd.GeoDataFrame(
        enw_df[
            [
                "pry_group",
                "firm_capacity_mva",
                "maximum_demand_mva_per_primary_substation",
                "geo_shape",
                "operator",
            ]
        ].rename(
            columns={
                "pry_group": "id",
                "maximum_demand_mva_per_primary_substation": "peak_demand_mva",
                "geo_shape": "geo_shape",
            }
        ),
        geometry="geo_shape",
        crs=CRS,
    )


def generate_npg_gdf() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for Northern Powergrid (NPg) primary substations and their service areas.

    The function processes substation data from NPg's heatmap demand data and Network Development
    Plan (NDP) datasets. It combines point locations of primary substations with their corresponding
    service area polygons and capacity/demand information based on NPg's Best View scenario.

    Returns:
        gpd.GeoDataFrame: DataFrame with EPSG:4326 (WGS84) projection containing:
            - id: Unique substation identifier
            - firm_capacity_mva: Maximum power the substation can safely deliver (MVA)
            - peak_demand_mva: Maximum observed power demand at the substation (MVA)
            - geo_shape: Polygon geometry representing the substation's service area
            - operator: Distribution network operator code ("NPg")
    """
    npg_substations = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["E_NPg_heatmap"]
    ).to_pandas()
    npg_demand = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["E_NPg_ndp_demand"]
    ).to_pandas()

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
    npg_demand["geo_shape"] = npg_demand["geo_shape"].apply(parse_binary_geometry)
    npg_shape = npg_demand["geo_shape"].drop_duplicates()

    # Create GeoDataFrames and perform spatial join
    npg_gdf = gpd.GeoDataFrame(npg, geometry="location", crs=CRS)
    npg_shape_gdf = gpd.GeoDataFrame(npg_shape, geometry="geo_shape", crs=CRS)
    npg_df = gpd.sjoin(npg_shape_gdf, npg_gdf, how="left", predicate="contains")
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
    Generate a GeoDataFrame for Scottish Power Energy Networks (SPEN) primary substations and their
    service areas.

    The function processes substation data from both SP Distribution (SPD) and SP Manweb (SPM)
    regions (Scotland and North Wales respectively), combining their respective heat maps and network development plan polygon datasets.
    It merges point locations of substations with their service area boundaries.

    Returns:
        gpd.GeoDataFrame: DataFrame with EPSG:4326 (WGS84) projection containing:
            - id: Unique substation identifier
            - firm_capacity_mva: Maximum power the substation can safely deliver (MVA)
            - peak_demand_mva: Maximum observed power demand at the substation (MVA)
            - geo_shape: Polygon geometry representing the substation's service area
            - operator: Distribution network operator code ("SPEN")
    """
    spen_spd_substations = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["S_SPEN_spd_substations"]
    ).to_pandas()
    spen_spm_substations = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["W_SPEN_spm_substations"]
    ).to_pandas()
    spen_spd_shape = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["S_SPEN_spd_polygons"]
    ).to_pandas()
    spen_spm_shape = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["W_SPEN_spm_polygons"]
    ).to_pandas()

    # Process SPM data
    # geometry data for SPM is only available at the primary group level which are groups of 1-3 primary substations
    # therefore we need to first aggregate primary substation data to primary group level
    spen_spm_substations = (
        spen_spm_substations.groupby("primary_group")[
            ["firm_capacity_mva", "maximum_load_mva"]
        ]
        .sum()
        .reset_index()
    )

    spm_df = spen_spm_substations.merge(spen_spm_shape, how="left", on="primary_group")
    spm_df = spm_df.rename(columns={"primary_group": "substation_name"})
    spm_df = spm_df[
        [
            "substation_name",
            "firm_capacity_mva",
            "maximum_load_mva",
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
            "geo_shape",
        ]
    ]

    # Combine SPM and SPD data
    spen_df = pd.concat([spd_df, spm_df])
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
    Generate a GeoDataFrame for Scottish and Southern Electricity Networks (SSEN) primary substations
    and their service areas.

    The function processes substation data from both SHEPD (Shetland) and SEPD (Scotland) regions.
    It combines demand heat map data with primary substation boundary polygons from both regions.

    Returns:
        gpd.GeoDataFrame: DataFrame with EPSG:4326 (WGS84) projection containing:
            - id: Unique substation identifier
            - firm_capacity_mva: Maximum power the substation can safely deliver (MVA)
            - peak_demand_mva: Maximum observed power demand at the substation (MVA)
            - geo_shape: Polygon geometry representing the substation's service area
            - operator: Distribution network operator code ("SSEN")
    """
    sepd_shape = gpd.read_file(config["data_source"]["S_SSEN_sepd_bounds"])
    shepd_shape = gpd.read_file(config["data_source"]["E_SSEN_shepd_bounds"])
    ssen_shape = pd.concat([shepd_shape, sepd_shape])

    ssen_demand = load_transform_df_ssen_demand().to_pandas()

    ssen_df = ssen_demand.merge(
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


def load_transform_df_ssen_demand() -> pl.DataFrame:
    """
    Load and transform demand data for Scottish and Southern Electricity Networks (SSEN) primary substations.

    The function processes substation data from SHEPD (Shetland), SEPD (Scotland), and South England regions.

    Returns:
        pl.DataFrame: dataset containing:
            - Primary Substation Name: Unique substation identifier
            - Nameplate rating (MVA): Maximum power the substation can safely deliver (MVA)
            - Maximum Load (MVA): Maximum observed power demand at the substation (MVA)
            - Latitude, Longitude, and Grid Reference geospatial information
    """
    cols = [
        "Primary Substation Name",
        "Nameplate rating (MVA)",
        "Maximum Load (MVA)",
        "Latitude",
        "Longitude",
        "Grid Reference",
    ]

    ssen_demand = []

    ssen_demand.append(
        base_getters.get_df_from_csv_s3_path(
            config["data_source"]["E_SSEN_demand"], columns=cols
        ).select(cols)
    )

    ssen_demand.append(
        base_getters.get_df_from_excel_s3_path(
            config["data_source"]["S_SSEN_demand"], sheet_name="PS", columns=cols
        ).select(cols)
    )

    ssen_demand.append(
        base_getters.get_df_from_excel_s3_path(
            config["data_source"]["SHET_SSEN_demand"], sheet_name="PS", columns=cols
        ).select(cols)
    )

    ssen_demand = pl.concat(ssen_demand).with_columns(
        pl.col("Primary Substation Name").str.to_uppercase()
    )

    return ssen_demand


def generate_ukpn_gdf() -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame for UK Power Networks (UKPN) primary substations and their service areas.

    The function processes primary substation data including seasonal capacity constraints and demand
    headroom. It calculates actual demand from percentage-based headroom values and combines this with
    service area polygons for each substation.

    Returns:
        gpd.GeoDataFrame: DataFrame with EPSG:4326 (WGS84) projection containing:
            - id: Unique substation identifier
            - firm_capacity_mva: Maximum power the substation can safely deliver (MVA)
            - peak_demand_mva: Maximum observed power demand at the substation (MVA)
            - geo_shape: Polygon geometry representing the substation's service area
            - operator: Distribution network operator code ("UKPN")
    """
    ukpn_primary_substations = base_getters.get_df_from_parquet_s3_path(
        config["data_source"]["E_UKPN_primaries"]
    ).to_pandas()

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
    Generate a GeoDataFrame for Western Power Distribution (WPD) primary substations and their
    service areas.

    The function processes network capacity data across four WPD regions: East Midlands, South Wales,
    South West, and West Midlands. It combines point locations and demand data from the network
    capacity map with service area boundary polygons from regional shapefiles.

    Returns:
        gpd.GeoDataFrame: DataFrame with EPSG:4326 (WGS84) projection containing:
            - id: Unique substation identifier
            - firm_capacity_mva: Maximum power the substation can safely deliver (MVA)
            - peak_demand_mva: Maximum observed power demand at the substation (MVA)
            - geo_shape: Polygon geometry representing the substation's service area
            - operator: Distribution network operator code ("WPD")
    """
    wpd_substations = base_getters.get_df_from_csv_s3_path(
        config["data_source"]["EW_WPD_capacity"]
    ).to_pandas()

    # Load shape files for each region
    wpd_shapes = []
    for region in [
        "E_WPD_east_midlands_bounds",
        "W_WPD_south_wales_bounds",
        "E_WPD_south_west_bounds",
        "E_WPD_west_midlands_bounds",
    ]:
        wpd_shapes.append(
            base_getters.get_gdf_from_gpkg_s3_path(config["data_source"][region])
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
