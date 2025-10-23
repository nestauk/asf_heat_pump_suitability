import geopandas as gpd
import pandas as pd

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


def transform_gdf_non_residential_buildings(
    important_building_gdf: gpd.GeoDataFrame,
    railway_station_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    building_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Transform important buildings, railway station, and points of interest data to create a dataframe of polygons representing buildings
    which are unlikely to contain residential properties, e.g. hospitals, train stations, museums etc. Important buildings
    and POI geopoints should include only types which are unlikely to be in mixed-use (residential and commercial) buildings.

    Args:
        important_building_gdf (gpd.GeoDataFrame): OS OpenMap Local important building footprints in area of interest
        railway_station_gdf (gpd.GeoDataFrame): OS OpenMap Local railway station point geometries in area of interest
        poi_gdf (gpd.GeoDataFrame): non-domestic Points of Interest geopoints in area of interest
        building_gdf (gpd.GeoDataFrame): all building footprints in area of interest

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
                building_gdf.crs,
            }
        )
        == 1
    )

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

    # Get buildings which are railway stations (railway stations are only given as point geometries)
    railway_station_gdf = building_gdf.sjoin(
        railway_station_gdf, how="inner", predicate="contains"
    )

    return pd.concat(
        [
            exclude_buildings_gdf[["geometry"]],
            railway_station_gdf[["geometry"]],
            poi_buildings_gdf[["geometry"]],
        ]
    )
