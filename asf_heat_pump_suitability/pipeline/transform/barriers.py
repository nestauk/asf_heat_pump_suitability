"""Functions to transform barrier data for clustering and other usages."""

from typing import Optional, List

import pandas as pd
import geopandas as gpd

from asf_heat_pump_suitability.getters import load_geodata


def load_transform_gdf_polygon_barriers(
    grid_squares: Optional[List[str]],
    road_types: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    """
    Load physical barriers with (Multi)Polygon geometries for the specified grid squares, these include:
    - Green space
    - Water bodies
    - Tidal boundaries
    - Woodland

    Additionally, load physical barriers with (Multi)LineString geometries - railways, and roads of the following types:
    "A Road", "B Road", "Motorway", "Minor Road"- for the specified grid squares. A buffer is added around each
    geometry to cover the width of the road / railway.

    Args:
        grid_squares (Optional[List[str]]): names of grid squares in OS mapping for regions of Great Britain to be loaded.
        Find grid square information at: https://www.ordnancesurvey.co.uk/documents/resources/guide-to-nationalgrid.pdf
        road_types (Optional[List[str]]): names of road type(s) which should act as barriers. Default to using all road types. Options:
        - Motorway
        - A Road
        - B Road
        - Minor Road
        - Local Road
        - Local Access Road
        - Restricted Local Access Road
        - Secondary Access Road

    Returns:
        gpd.GeoDataFrame: physical barriers with (Multi)Polygon geometries
    """
    # Polygons
    forest_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="woodland", grid_squares=grid_squares
    )

    greenspace_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="greenspace_site", grid_squares=grid_squares
    )

    surface_water_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="surface_water_area", grid_squares=grid_squares
    )

    tidal_water_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="tidal_water", grid_squares=grid_squares
    )

    # Linestrings
    barrier_roads_gdf = load_geodata.load_gdf_os_openroad(grid_squares=grid_squares)
    if road_types:
        road_types = [road_type.lower() for road_type in road_types]
        barrier_roads_gdf = barrier_roads_gdf[
            barrier_roads_gdf["function"].str.lower().isin(road_types)
        ]

    railways_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="railway_track", grid_squares=grid_squares
    )

    line_overlays = [barrier_roads_gdf, railways_gdf]
    line_overlay_gdf = pd.concat([gdf[["geometry"]] for gdf in line_overlays])

    # TODO make more specific for different road types
    # Add buffer assumed to be width of road / railway (3.5m total - 1.75m either side)
    line_overlay_gdf["geometry"] = line_overlay_gdf.geometry.buffer(1.75)

    overlays = [
        forest_gdf,
        greenspace_gdf,
        tidal_water_gdf,
        surface_water_gdf,
        line_overlay_gdf,
    ]

    return pd.concat([gdf[["geometry"]] for gdf in overlays])
