"""
Functions to transform data related to identifying non-domestic buildings.
"""

import geopandas as gpd
import pandas as pd

from asf_heat_pump_suitability.pipeline.transform import uprns

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


def generate_gdf_non_residential_buildings(
    important_building_gdf: gpd.GeoDataFrame,
    railway_station_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    poi_unfiltered_gdf: gpd.GeoDataFrame,
    building_gdf: gpd.GeoDataFrame,
    uprns_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Use important buildings, railway station, and points of interest data to create a dataframe of polygons representing buildings
    which are unlikely to contain residential properties, e.g. hospitals, train stations, museums etc. Important buildings
    and POI geopoints should include only types which are unlikely to be in mixed-use (residential and commercial) buildings.

    Args:
        important_building_gdf (gpd.GeoDataFrame): OS OpenMap Local important building footprints in area of interest
        railway_station_gdf (gpd.GeoDataFrame): OS OpenMap Local railway station point geometries in area of interest
        poi_gdf (gpd.GeoDataFrame): non-domestic Points of Interest geopoints in area of interest
        poi_unfiltered_gdf (gpd.GeoDataFrame): All Points of Interest geopoints in area of interest
        building_gdf (gpd.GeoDataFrame): all building footprints in area of interest
        uprns_gdf (gpd.GeoDataFrame): UPRNs with point geometries in area of interest

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
                poi_gdf.crs,
                poi_unfiltered_gdf.crs,
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
    poi_buildings_gdf = building_gdf.sjoin(poi_gdf, how="inner", predicate="contains")
    poi_unfiltered_gdf = building_gdf.sjoin(
        poi_unfiltered_gdf, how="inner", predicate="contains"
    ).drop("index_right", axis=1)

    # Get buildings which are railway stations (railway stations are only given as point geometries)
    railway_station_gdf = railway_station_gdf.sjoin(
        building_gdf, how="inner", predicate="within"
    )

    exclude_buildings_gdf = pd.concat(
        [
            exclude_buildings_gdf[["geometry"]],
            railway_station_gdf[["geometry"]],
            poi_buildings_gdf[["geometry"]],
        ]
    )

    # creating list of buildings from all important buildings and POI data with exactly 1 UPRN in them

    # join unfiltered important buildings and POI with all UPRNs
    important_building_gdf = important_building_gdf.sjoin(
        uprns_gdf, how="left", predicate="contains"
    ).drop("index_right", axis=1)
    poi_unfiltered_gdf = poi_unfiltered_gdf.sjoin(
        uprns_gdf, how="left", predicate="contains"
    ).drop("index_right", axis=1)

    # find number of UPRNs per building
    important_building_gdf = (
        important_building_gdf.groupby("geometry").size().reset_index(name="UPRN_count")
    )
    poi_unfiltered_gdf = (
        poi_unfiltered_gdf.groupby("geometry").size().reset_index(name="UPRN_count")
    )

    # select buildings with 1 UPRN
    important_building_gdf = important_building_gdf.loc[
        important_building_gdf["UPRN_count"] == 1
    ]
    poi_unfiltered_gdf = poi_unfiltered_gdf.loc[poi_unfiltered_gdf["UPRN_count"] == 1]

    # add this to list of buildings to exclude
    exclude_buildings_gdf = pd.concat(
        [
            exclude_buildings_gdf[["geometry"]],
            important_building_gdf[["geometry"]],
            poi_unfiltered_gdf[["geometry"]],
        ]
    )

    # Get locations of UPRNs in domestic EPC
    include_uprns = uprns.load_set_valid_epc_uprns(epc_type="domestic")
    uprns_gdf = uprns_gdf[uprns_gdf["UPRN"].isin(include_uprns)].copy()

    # Find buildings that contain a domestic UPRN
    exclude_buildings_gdf = exclude_buildings_gdf.sjoin(
        uprns_gdf[["UPRN", "geometry"]], how="left", predicate="contains"
    )

    # Filter to buildings that don't contain a domestic UPRN
    exclude_buildings_gdf = exclude_buildings_gdf[
        exclude_buildings_gdf["UPRN"].isnull()
    ].drop(columns=["UPRN", "index_right"])

    # Normalize to drop duplicate building footprints
    exclude_buildings_gdf["geometry"] = exclude_buildings_gdf.normalize()
    return exclude_buildings_gdf.drop_duplicates()
