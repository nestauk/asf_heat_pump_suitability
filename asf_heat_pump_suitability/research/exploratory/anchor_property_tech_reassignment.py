# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
import matplotlib.pyplot as plt
import shapely
import folium
import os
from asf_heat_pump_suitability import PROJECT_DIR, config
from asf_heat_pump_suitability.getters import load_tree_input

# %%
# load datasets
# convert to CRS 27700 if necessary
building_footprints = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="building", grid_squares="SX"
)
anchor_properties = gpd.read_file(
    "s3://asf-heat-pump-suitability/dump/hack_day/anchor_properties_gb.geojson"
).to_crs(27700)
tech_types = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_building_most_suitable_tech.geojson"
).to_crs(27700)
important_buildings = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="important_building", grid_squares="SX"
)

# %%
# categories in important building list that could be anchors
anchor_categories = [
    "Primary Education",
    "Museum",
    "Library",
    "Further Education",
    "Secondary Education",
    "Fire Station",
    "Sports And Leisure Centre",
    "Hospital",
    "Higher or University Education",
    "Special Needs Education",
    "Medical Care Accommodation",
    "Non State Primary Education",
    "Non State Secondary Education",
    "Art Gallery",
    "Police Station",
    "Hospice",
    "Airport",
]

# %%
# change all solutions to communal, except district heat network

tech_mapping = {
    "Individual solution": "Communal solutions",
    "District heat network": "District heat network",
    "Individual solution or Networked GSHP": "Communal solutions",
    "Networked GSHP": "Communal solutions",
    "Communal solutions": "Communal solutions",
    "Individual solution or District heat network": "Communal solutions",
}

# %%
# for plotting

COLOURS = {
    "Individual solution": "#18A48C",
    "District heat network": "#EA2541",
    "Individual solution or Networked GSHP": "grey",
    "Networked GSHP": "#0000FF",
    "Communal solutions": "#FF6E47",
    "Individual solution or District heat network": "gray",
}

# %%
# select anchors out of important building gdf using anchor_categories list
important_buildings_filtered = important_buildings[
    important_buildings["CLASSIFICA"].isin(anchor_categories)
]

# %%
# add building footprint data to POI anchor properties so geometry isn't just a point
anchors_with_footprint = (
    building_footprints.sjoin(anchor_properties, how="inner", predicate="contains")
).drop("index_right", axis=1)

# %%
# add POI and important building lists together and remove duplicate buildings
# I don't think the removing duplicates is fully working, need to check
all_anchors_with_footprint = pd.concat(
    [anchors_with_footprint, important_buildings_filtered]
)
all_anchors_with_footprint = all_anchors_with_footprint.drop_duplicates(["geometry"])

# %%
# function to find distance between a residential building and the nearest anchor + resassign tech type based on this distance


def assign_communal_solutions(
    tech_gdf: gpd.GeoDataFrame, anchors_gdf: gpd.GeoDataFrame, radius: float
) -> gpd.GeoDataFrame:
    """
    Finds nearest anchors and reassigns tech type to 'Communal' if within radius.
    """
    # Preserve original geometry for distance calculation after join
    anchors_gdf["anchor_geometry"] = anchors_gdf.geometry

    # Spatial join to find nearest anchor for every building
    nearest = tech_gdf.sjoin_nearest(anchors_gdf, how="left")
    nearest["distance"] = nearest["geometry"].distance(nearest["anchor_geometry"])

    # Apply reassignment logic
    reassigned_tech = (nearest[nearest["distance"] < radius]).replace(
        tech_mapping, inplace=False
    )
    # save previous tech type for visualisation
    reassigned_tech["old_tech"] = nearest[nearest["distance"] < radius][
        "1st_most_suitable_solution"
    ]

    return reassigned_tech


# %%
# set anchor_radius as the radius around the anchor you want to change to communal
anchor_radius = 100
reassigned_tech = assign_communal_solutions(
    tech_types, all_anchors_with_footprint, anchor_radius
)

# %%
reassigned_tech.head()


# %%
def dissolve_techs_and_plot_folium(
    reassigned_tech: gpd.GeoDataFrame,
    anchor_properties_with_footprint: gpd.GeoDataFrame,
    important_building_anchors: gpd.GeoDataFrame,
    colours: dict = COLOURS,
):
    """
    Plot the resulting polygons together for each tech type in Folium.

    Args:
        gdf (gpd.GeoDataFrame): dataframe with polygons to dissolve and plot and tech labels
        colours (dict): tech type labels and their corresponding colours for plotting

    Returns Folium map
    """
    # Dissolve by tech type
    dissolved_gdf = reassigned_tech.dissolve(
        by="1st_most_suitable_solution"
    ).reset_index()
    dissolved_gdf["colour"] = dissolved_gdf["1st_most_suitable_solution"].map(colours)

    # Convert dissolved tech polygons and buildings to EPSG 4326 for plotting
    dissolved_4326_gdf = dissolved_gdf.to_crs(epsg=4326)
    anchor_4326 = anchor_properties_with_footprint.to_crs(4326)
    important_4326 = important_building_anchors.to_crs(4326)

    # Get centre of boundary to centre map
    boundary_4326 = dissolved_4326_gdf["geometry"].values[0]
    centre_map = shapely.get_coordinates(boundary_4326.centroid)

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    for _, r in dissolved_4326_gdf.iterrows():
        colour = colours[r["1st_most_suitable_solution"]]
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x, colour=colour: {
                "fillColor": colour,
                "weight": 0.1,
                "fillOpacity": 0.66,
            },
        )
        folium.Popup(r["old_tech"]).add_to(geo_j)
        geo_j.add_to(m)

    for _, r in anchor_4326.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x: {
                "color": "blue",
                "weight": 0.1,
                "fillOpacity": 0.66,
            },
        )
        folium.Popup(r["main_category"]).add_to(geo_j)
        geo_j.add_to(m)

    for _, r in important_4326.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x: {
                "color": "purple",
                "weight": 0.1,
                "fillOpacity": 0.66,
            },
        )
        folium.Popup(r["CLASSIFICA"]).add_to(geo_j)
        geo_j.add_to(m)

    return m


# %%
# click on the building to see either building type or previous tech type
dissolve_techs_and_plot_folium(
    reassigned_tech, anchors_with_footprint, important_buildings_filtered, COLOURS
)

# %%
