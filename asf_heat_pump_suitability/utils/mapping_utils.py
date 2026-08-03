"""
Util functions for plotting geospatial data in maps.
"""

import folium
import shapely
from shapely.ops import transform
import pyproj
import geopandas as gpd
from typing import Optional, Dict
from pathlib import Path
import os

from asf_heat_pump_suitability import config

TECH_TYPES = dict(sorted(config["constant"]["tech_types"].items()))
COLOUR_MAPPING = dict(sorted(config["constant"]["tech_colour_mapping"].items()))


def plot_folium_cluster_map(
    clusters_gdf: gpd.GeoDataFrame,
    popup_col: Optional[str] = None,
    save_as: Optional[str] = None,
) -> folium.Map:
    """
    Plot cluster polygons on a satellite folium map.

    Args:
        clusters_gdf (gpd.GeoDataFrame): geodataframe containing (Multi)Polygons to plot.
        popup_col (str): name of column containing information for popup. Optional.
        save_as (str): file name to save as, without file extension, e.g. `map_01`. Saves a local html file copy to /outputs/maps/.

    Returns:
        Folium.Map
    """
    # Reproject clusters to 4326
    map_gdf = clusters_gdf.to_crs(epsg=4326)

    # Get centre of boundary to centre map
    centre_map = shapely.get_coordinates(
        map_gdf["geometry"].make_valid().union_all().centroid
    )

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    colour_mapping = dict(zip(TECH_TYPES.values(), COLOUR_MAPPING.values()))

    for tech, colour in colour_mapping.items():
        _map_gdf = map_gdf[map_gdf["assigned_tech"] == tech]
        _plot_folium_polygons(m, _map_gdf, popup_col, colour)

    if save_as:
        PROJECT_DIR = Path(__file__).resolve().parents[2]
        file_path = os.path.join(PROJECT_DIR, "outputs", "maps", f"{save_as}.html")
        m.save(file_path)

    return m


def plot_folium_polygon_map(
    boundary: shapely.Polygon | shapely.MultiPolygon,
    gdf_dict: dict[str, gpd.GeoDataFrame],
    colour_mapping: Dict[str, str],
    popup_col: Optional[str] = None,
    save_as: Optional[str] = None,
    crs: int = 27700,
) -> folium.Map:
    """
    Plot polygons on a satellite folium map.

    Args:
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary of map area
        gdf_dict (dict[str, gpd.GeoDataFrame]): dictionary containing label key and corresponding geodataframe value(s) containing (Multi)Polygons to plot.
        colour_mapping (dict[str, str]): dictionary containing label key (corresponding to each geodataframe in `gdf_dict`), with a string colour value.
        popup_col (str): name of column containing information for popup. Optional.
        save_as (str): file name to save as. Saves a local html file copy to /outputs/maps/.
        crs (int): CRS of `polygon_gdf`. Default 27700.

    Returns:
        Folium.Map
    """
    # Create transformer for reprojection
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init=f"epsg:{crs}"),
        pyproj.Proj(init="epsg:4326"),
    )

    # Reproject boundary to 4326
    map_boundary = transform(project.transform, boundary)
    # Get centre of boundary to centre map
    centre_map = shapely.get_coordinates(map_boundary.centroid)

    # Reproject polygons to 4326
    map_gdfs = {k: gdf.to_crs(epsg=4326) for k, gdf in gdf_dict.items()}

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    for k, map_gdf in map_gdfs.items():
        _plot_folium_polygons(m, map_gdf, popup_col, colour_mapping[k])

    if save_as:
        PROJECT_DIR = Path(__file__).resolve().parents[2]
        file_path = os.path.join(PROJECT_DIR, "outputs", "maps", f"{save_as}.html")
        m.save(file_path)

    return m


def _plot_folium_polygons(
    m: folium.Map, map_gdf: gpd.GeoDataFrame, popup_col: str, colour: str
) -> folium.Map:
    """
    Plot polygons on a folium map.

    Args:
        m (folium.Map): map object to plot on
        map_gdf (gpd.GeoDataFrame): GeoDataFrame containing (Multi)Polygons to plot.
        popup_col (str): name of column containing information for popup. Optional.
        colour (str): colour of polygons to plot

    Returns:
        folium.Map
    """
    # Plot polygons with pop up info
    if popup_col:
        for _, r in map_gdf.iterrows():
            sim_geo = gpd.GeoSeries(r["geometry"])
            geo_j = sim_geo.to_json()
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour=colour: {
                    "fillColor": colour,
                    "fillOpacity": 0.3,
                    "color": colour,
                },
            )
            folium.Popup(f"{popup_col}: {r[popup_col]}").add_to(geo_j)
            geo_j.add_to(m)
    # Plot polygons without popup info
    else:
        for _, r in map_gdf.iterrows():
            sim_geo = gpd.GeoSeries(r["geometry"])
            geo_j = sim_geo.to_json()
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour=colour: {
                    "fillColor": colour,
                    "fillOpacity": 0.3,
                    "color": colour,
                },
            )
            geo_j.add_to(m)

    return m
