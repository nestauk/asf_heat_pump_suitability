# %% [markdown]
# ## Anchor Property Tech Reassignment
#
# notebook to find buildings within a specified distance of an anchor proprty, reassign their tech type as communal (unless assigned district heat network) and visualise the results

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

# anchor properties list contains POI buildings filtered by building type, see below for list of categories
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
# for reference, these are all the anchor property categories in the POI based list
anchor_properties["main_category"].unique()

# %%
# these are all possible building type categories in the OS important building layer
important_buildings["CLASSIFICA"].unique()

# %%
# now filter to categories in important building layer that could be anchor properties
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
all_anchors_with_footprint = pd.concat(
    [anchors_with_footprint, important_buildings_filtered]
)
all_anchors_with_footprint["geometry"] = all_anchors_with_footprint.normalize()
all_anchors_with_footprint = all_anchors_with_footprint.drop_duplicates(["geometry"])

# %%
# function to find distance between a residential building and the nearest anchor + resassign tech type based on this distance


def assign_communal_solutions(
    tech_gdf: gpd.GeoDataFrame, anchors_gdf: gpd.GeoDataFrame, radius: float
) -> gpd.GeoDataFrame:
    """
    Finds nearest anchors and reassigns tech type to 'Communal' if within radius.
    Args:
        tech_gdf (gpd.GeodataFrame): dataframe with most suitable tech types assigned to residential buildings
        anchors_gdf (gpd.GeodataFrame): dataframe with anchor properties and their building footprints
        radius (float): max distance from anchor property to reassign tech types
    Returns:
        gpd.GeoDataFrame: dataframe of residential buildings that have had their tech type reassigned, with columns for old tech type, corresponding anchor property geometry and distance from anchor added

    """

    # Spatial join to find nearest anchor for every building
    reassigned_tech = tech_gdf.sjoin_nearest(
        anchors_gdf, how="left", max_distance=radius, distance_col="distance_m"
    )

    # save old tech type in a column
    reassigned_tech["old_tech"] = reassigned_tech["1st_most_suitable_solution"]

    # replace tech types for buildings < max radius away from an anchor
    reassigned_tech = (
        reassigned_tech[(reassigned_tech["distance_m"]).notna()]
    ).replace({"1st_most_suitable_solution": tech_mapping})

    return reassigned_tech


# %%
# set anchor_radius as the radius around the anchor you want to change to communal
anchor_radius = 50
reassigned_tech = assign_communal_solutions(
    tech_gdf=tech_types, anchors_gdf=all_anchors_with_footprint, radius=anchor_radius
)

# %%
reassigned_tech.head()


# %%
def dissolve_techs_and_plot_folium(
    reassigned_tech: gpd.GeoDataFrame,
    anchor_properties_with_footprint: gpd.GeoDataFrame,
    important_building_anchors: gpd.GeoDataFrame,
    tech_to_plot: str,
    colours: dict = COLOURS,
):
    """
    Plot the resulting polygons together for each tech type in Folium.

    Args:
        reassigned_tech (gpd.GeoDataFrame): dataframe with polygons to dissolve and plot and tech labels
        anchor_properties_with_footprints (gpd.GeoDataFrame): dataframe with polygons of anchor properties from POI data
        important_building_anchors (gpd.GeoDataFrame): dataframe with polygons of anchor properties from important buildings lists
        tech_to_plot (str): choose to plot either reassigned or old tech types
        colours (dict): tech type labels and their corresponding colours for plotting

    Returns Folium map
    """
    # Dissolve by tech type
    dissolved_gdf = reassigned_tech.dissolve(by=tech_to_plot).reset_index()
    dissolved_gdf["colour"] = dissolved_gdf[tech_to_plot].map(colours)

    # Convert dissolved tech polygons and buildings to EPSG 4326 for plotting
    dissolved_4326_gdf = dissolved_gdf.to_crs(epsg=4326)
    anchor_4326 = anchor_properties_with_footprint.to_crs(4326)
    important_4326 = important_building_anchors.to_crs(4326)

    # Get centre of first polygon in dissolved gdf to centre map
    centre_4326 = dissolved_4326_gdf["geometry"].values[0]
    centre_map = shapely.get_coordinates(centre_4326.centroid)

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    for _, r in dissolved_4326_gdf.iterrows():
        colour = colours[r[tech_to_plot]]
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
                "color": "purple",
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
    reassigned_tech=reassigned_tech,
    anchor_properties_with_footprint=anchors_with_footprint,
    important_building_anchors=important_buildings_filtered,
    tech_to_plot="1st_most_suitable_solution",
    colours=COLOURS,
)

# %%
# just buildings that are assigned 'communal'
mask = (reassigned_tech["distance_m"].notna()) & (
    reassigned_tech["1st_most_suitable_solution"] == "Communal solutions"
)
dissolve_techs_and_plot_folium(
    reassigned_tech=reassigned_tech[mask],
    anchor_properties_with_footprint=anchors_with_footprint,
    important_building_anchors=important_buildings_filtered,
    tech_to_plot="1st_most_suitable_solution",
    colours=COLOURS,
)

# %%
dissolve_techs_and_plot_folium(
    reassigned_tech=reassigned_tech[mask],
    anchor_properties_with_footprint=anchors_with_footprint,
    important_building_anchors=important_buildings_filtered,
    tech_to_plot="old_tech",
    colours=COLOURS,
)

# %% [markdown]
# ## Value counts and proportion data

# %%
# all buildings that have been reassigned as communal- what tech type did they have before?
(
    reassigned_tech[
        reassigned_tech["1st_most_suitable_solution"] != reassigned_tech["old_tech"]
    ]
)["old_tech"].value_counts(dropna=False)

# %%
proportion_reassigned = (
    len(
        reassigned_tech[
            reassigned_tech["1st_most_suitable_solution"] != reassigned_tech["old_tech"]
        ]
    )
    / len(tech_types)
) * 100
print(
    f"Proportion of buildings which have been reassigned tech type to communal: {proportion_reassigned:.2f}%"
)

# %%
