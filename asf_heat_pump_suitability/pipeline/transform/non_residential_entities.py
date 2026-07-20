"""
Functions to transform data related to identifying non-domestic buildings.
"""

import geopandas as gpd
import pandas as pd
import numpy as np

# TODO these may need refinement for large cities where overlap with residential is possible
NO_RESIDENTIAL_OVERLAP_BUILDING_TYPES = [
    "Port Consisting of Docks and Nautical Berthing",
    "Fire Station",
    "Hospital",
    "Non State Secondary Education",
    "Secondary Education",
    "Higher or University Education",
    "Primary Education",
    "Place Of Worship",
    "Medical Care Accommodation",
    "Museum",
    "Special Needs Education",
    "Further Education",
    "Non State Primary Education",
    "Coach Station",
    "Police Station",
    "Sports And Leisure Centre",
    "Vehicular Ferry Terminal",
    "Hospice",
    "Bus Station",
    "Road User Services",
    "Passenger Ferry Terminal",
]


def _generate_gdf_fully_non_domestic_buildings(
    important_building_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    uprns_gdf: gpd.GeoDataFrame,
    building_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Function to create dataframe of buildings from the important buildings with exactly 1 UPRN in them, and buildings from POI data with equal number of POI and UPRNs. These are unlikely to be residential buildings
    Args:
        important_building_gdf (gpd.GeoDataFrame): OS OpenMap Local important building footprints in area of interest
        poi_gdf (gpd.GeoDataFrame): All Points of Interest geopoints in area of interest
        uprns_gdf (gpd.GeoDataFrame): all UPRNs with point geometries in area of interest
        building_gdf (gpd.GeoDataFrame): all building footprints in area of interest
    Returns:
        gpd.GeoDataFrame: geometries of buildings which are in the important buildings list and have 1 UPRN in them, or in the POI list and have same number of UPRNs and POI
    """
    # join POI data to building footprints
    poi_buildings_gdf = building_gdf.sjoin(
        poi_gdf, how="inner", predicate="contains"
    ).drop("index_right", axis=1)

    # count instances where multiple POI are located inside a building and convert result to gdf
    poi_buildings_df = (
        poi_buildings_gdf.groupby("ID")
        .agg(POI_count=("ID", "count"), geometry=("geometry", "first"))
        .reset_index()
    )
    poi_buildings_gdf = gpd.GeoDataFrame(
        poi_buildings_df,
        geometry=poi_buildings_df["geometry"],
        crs=poi_buildings_gdf.crs,
    )

    # join important buildings and POI with all UPRNs
    important_building_gdf = important_building_gdf.sjoin(
        uprns_gdf, how="left", predicate="contains"
    ).drop("index_right", axis=1)

    poi_buildings_gdf = poi_buildings_gdf.sjoin(
        uprns_gdf, how="left", predicate="contains"
    ).drop("index_right", axis=1)

    # find number of UPRNs per building for important buildings and convert result to gdf
    important_building_df = (
        important_building_gdf.groupby("ID")
        .agg(UPRN_count=("UPRN", "count"), geometry=("geometry", "first"))
        .reset_index()
    )
    important_building_gdf = gpd.GeoDataFrame(
        important_building_df,
        geometry=important_building_df["geometry"],
        crs=important_building_gdf.crs,
    )

    # find number of UPRNs per building for POI buildings and convert result to gdf
    poi_buildings_df = (
        poi_buildings_gdf.groupby("ID")
        .agg(
            UPRN_count=("UPRN", "count"),
            POI_count=("POI_count", "first"),
            geometry=("geometry", "first"),
        )
        .reset_index()
    )
    poi_buildings_gdf = gpd.GeoDataFrame(
        poi_buildings_df,
        geometry=poi_buildings_df["geometry"],
        crs=poi_buildings_gdf.crs,
    )

    # select building footprints from important building data where UPRN count = 1
    important_building_gdf = important_building_gdf.loc[
        important_building_gdf["UPRN_count"] == 1
    ]

    # select building footprints from POI data where UPRN count = POI count
    poi_buildings_gdf = poi_buildings_gdf.loc[
        poi_buildings_gdf["UPRN_count"] <= poi_buildings_gdf["POI_count"]
    ]

    # concat gdfs and keep only geometry column
    non_domestic_buildings_gdf = pd.concat(
        [important_building_gdf[["geometry"]], poi_buildings_gdf[["geometry"]]]
    )

    return non_domestic_buildings_gdf


def generate_gdf_non_residential_buildings(
    important_building_gdf: gpd.GeoDataFrame,
    railway_station_gdf: gpd.GeoDataFrame,
    non_domestic_poi_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    building_gdf: gpd.GeoDataFrame,
    uprns_gdf: gpd.GeoDataFrame,
    domestic_epc_uprns: np.array,
) -> gpd.GeoDataFrame:
    """
    Use important buildings, railway station, and points of interest data to create a dataframe of polygons representing buildings
    which are unlikely to contain residential properties, e.g. hospitals, train stations, museums etc. Important buildings
    and POI geopoints should include only types which are unlikely to be in mixed-use (residential and commercial) buildings.

    Args:
        important_building_gdf (gpd.GeoDataFrame): OS OpenMap Local important building footprints in area of interest
        railway_station_gdf (gpd.GeoDataFrame): OS OpenMap Local railway station point geometries in area of interest
        non_domestic_poi_gdf (gpd.GeoDataFrame): non-domestic Points of Interest geopoints in area of interest, unlikely to be in mixed-use buildings
        poi_gdf (gpd.GeoDataFrame): All Points of Interest geopoints in area of interest
        building_gdf (gpd.GeoDataFrame): all building footprints in area of interest
        uprns_gdf (gpd.GeoDataFrame): all UPRNs with point geometries in area of interest
        domestic_epc_uprns (np.array): UPRNs in domestic EPC register

    Returns:
        gpd.GeoDataFrame: geometries of buildings which are unlikely to contain residential properties
    """
    print("Creating non-residential buildings dataset...")
    # Assert all gdfs have the same CRS
    assert (
        len(
            {
                important_building_gdf.crs,
                railway_station_gdf.crs,
                non_domestic_poi_gdf.crs,
                poi_gdf.crs,
                building_gdf.crs,
                uprns_gdf.crs,
            }
        )
        == 1
    ), "All GeoDataFrame inputs must have the same CRS"

    # Find important building classification column name
    col = None
    for name in ["CLASSIFICA", "classification"]:
        if name in important_building_gdf.columns:
            col = name
            break
    if not col:
        raise ValueError(
            "Important Building GeoDataFrame does not have a recognised building classification column (required)."
        )

    # Get buildings which are unlikely to have residential overlap
    exclude_buildings_gdf = important_building_gdf[
        important_building_gdf[col].isin(NO_RESIDENTIAL_OVERLAP_BUILDING_TYPES)
    ]
    non_domestic_poi_gdf = building_gdf.sjoin(
        non_domestic_poi_gdf, how="inner", predicate="contains"
    )

    # Get buildings which are railway stations (railway stations are only given as point geometries)
    railway_station_gdf = railway_station_gdf.sjoin(
        building_gdf, how="inner", predicate="within"
    )

    # creating list of buildings from all important buildings and POI data with exactly 1 UPRN in them- these are likely to be fully commercial units
    fully_non_domestic_buildings_gdf = _generate_gdf_fully_non_domestic_buildings(
        important_building_gdf=important_building_gdf,
        poi_gdf=poi_gdf,
        uprns_gdf=uprns_gdf,
        building_gdf=building_gdf,
    )

    # add this to list of buildings to exclude
    exclude_buildings_gdf = pd.concat(
        [
            exclude_buildings_gdf[["geometry"]],
            railway_station_gdf[["geometry"]],
            non_domestic_poi_gdf[["geometry"]],
            fully_non_domestic_buildings_gdf[["geometry"]],
        ]
    )

    # Normalize to drop duplicate building footprints
    exclude_buildings_gdf["geometry"] = exclude_buildings_gdf.normalize()
    exclude_buildings_gdf = exclude_buildings_gdf.drop_duplicates(subset=["geometry"])

    # Get locations of UPRNs in domestic EPC
    uprns_gdf = uprns_gdf[uprns_gdf["UPRN"].isin(domestic_epc_uprns)].copy()

    # Find buildings that contain a domestic UPRN
    exclude_buildings_gdf = exclude_buildings_gdf.sjoin(
        uprns_gdf[["UPRN", "geometry"]], how="left", predicate="contains"
    )

    # Filter to buildings that don't contain a domestic UPRN
    exclude_buildings_gdf = exclude_buildings_gdf[
        exclude_buildings_gdf["UPRN"].isnull()
    ].drop(columns=["UPRN", "index_right"])

    return exclude_buildings_gdf
