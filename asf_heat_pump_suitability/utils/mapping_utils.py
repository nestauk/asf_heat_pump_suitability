import folium
import shapely
from shapely.ops import transform
import pyproj
import geopandas as gpd
from typing import Optional
from pathlib import Path
import os


def plot_folium_polygon_map(
    polygon_gdf: gpd.GeoDataFrame,
    boundary: shapely.Polygon | shapely.MultiPolygon,
    popup_col: Optional[str],
    save_as: Optional[str],
    crs: int = 27700,
):
    """
    Plot polygons on a satellite folium map.

    Args:
        polygon_gdf (gpd.GeoDataFrame): geodataframe containing (Multi)Polygons to plot
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary of map area
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
    map_gdf = polygon_gdf.to_crs(epsg=4326)

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    # Plot building footprints
    if popup_col:
        for _, r in map_gdf.iterrows():
            sim_geo = gpd.GeoSeries(r["geometry"])
            geo_j = sim_geo.to_json()
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="blue": {
                    "fillColor": colour,
                    "fillOpacity": 0.3,
                },
            )
            folium.Popup(r[popup_col]).add_to(geo_j)
            geo_j.add_to(m)
    else:
        for _, r in map_gdf.iterrows():
            sim_geo = gpd.GeoSeries(r["geometry"])
            geo_j = sim_geo.to_json()
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="blue": {
                    "fillColor": colour,
                    "fillOpacity": 0.3,
                },
            )
            geo_j.add_to(m)

    if save_as:
        PROJECT_DIR = Path(__file__).resolve().parents[2]
        file_path = os.path.join(PROJECT_DIR, "outputs", "maps", f"{save_as}.html")
        m.save(file_path)

    return m
