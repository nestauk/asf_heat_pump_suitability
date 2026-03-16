"""
Functions to load specific raw datasets used in decision tree pipeline using base getters and sources in config. No/minimal preprocessing occurs in these functions.
"""

import geopandas as gpd
import pyogrio

from asf_heat_pump_suitability import config


def load_openmap_local_layer(
    layer: str,
    lad_boundary: gpd.GeoDataFrame,
    buffer_m: int = 1000,
) -> gpd.GeoDataFrame:
    """Load one OS OpenMap Local layer clipped to the buffered LAD boundary.

    Uses pyogrio's ``mask`` parameter to read only features that intersect the
    buffered boundary, avoiding the need for grid-square configuration.

    The 1 km buffer ensures that buildings straddling the LAD boundary are captured.

    Args:
        layer: Layer name as it appears in the geopackage (e.g. ``"building"``).
        lad_boundary: Single-row GeoDataFrame of the LAD boundary in EPSG:27700.
        buffer_m: Buffer distance in metres. Default 1000.

    Layer options:
        'building', 'car_charging_point', 'electricity_transmission_line',
        'foreshore', 'functional_site', 'glasshouse', 'important_building',
        'motorway_junction', 'named_place', 'railway_station', 'railway_track',
        'railway_tunnel', 'road', 'road_tunnel', 'roundabout',
        'surface_water_area', 'surface_water_line', 'tidal_boundary',
        'tidal_water', 'woodland'

    Returns:
        gpd.GeoDataFrame: OS OpenMap Local geometries for the specified layer,
        clipped to the buffered LAD boundary. CRS British National Grid (27700).
    """
    mask = lad_boundary.geometry.buffer(buffer_m).union_all()
    print(f"Loading OS OpenMap Local - {layer.title()} (masked to LAD boundary)...")
    return pyogrio.read_dataframe(
        config["inputs"]["geodata"]["os_openmap_local"],
        layer=layer,
        mask=mask,
    ).drop_duplicates(subset="ID")


def load_gdf_poi() -> gpd.GeoDataFrame:
    """
    Load and process Points of Interest data. CRS EPSG 4326.

    Returns:
        gpd.GeoDataFrame: Processed POI data containing types of POI specified

    Raises:
        ValueError: If required columns are missing
    """
    print("Loading POI data...")

    required_columns = [
        "id",
        "country",
        "main_category",
        "alternate_category",
        "geometry",
    ]
    poi = gpd.read_file(
        filename=config["inputs"]["geodata"]["poi"],
        columns=required_columns,
        layer="poi_uk",
    ).to_crs("EPSG:4326")
    print(f"POI CRS: {poi.crs}")
    return poi
