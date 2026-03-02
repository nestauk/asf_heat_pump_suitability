# %% [markdown]
# ## Domestic Filtering Exploration
#

# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
import matplotlib.pyplot as plt
import shapely
import folium
import os

# %%
# local imports
from asf_heat_pump_suitability import PROJECT_DIR
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import (
    load_tree_input,
    load_geodata,
    load_boundaries,
)
from asf_heat_pump_suitability.pipeline.transform import uprns, poi

# %%
# load datasets
building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="building", grid_squares="SX"
)
anchor_properties_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/dump/hack_day/anchor_properties_gb.geojson"
).to_crs(27700)
tech_types_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_building_most_suitable_tech.geojson"
).to_crs(27700)
important_buildings_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="important_building", grid_squares="SX"
)
epc_domestic_set = uprns.load_set_valid_epc_uprns(epc_type="domestic")
epc_commercial_set = uprns.load_set_valid_epc_uprns(epc_type="commercial")
uprns_df = load_geodata.load_df_osopen_uprn()
uprns_GB_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

# %%
# Load POI data

poi_gdf = load_tree_input.load_gdf_poi()
poi_gdf = poi.transform_gdf_poi(
    poi_gdf,
    filter_categories=poi.load_set_non_domestic_poi_categories(),
)

# %% [markdown]
# ## EPC Data - Properties that should be domestic

# %%
# gdf of all uprns in GB that are in the domestic EPC data

uprns_GB_gdf = uprns_GB_gdf[(uprns_GB_gdf["UPRN"].isin(epc_domestic_set))]

# %%
# filter these to plymouth only

la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
    select_las="Plymouth"
)
domestic_uprns_gdf = uprns_GB_gdf.sjoin(
    la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
    how="inner",
    predicate="intersects",
).drop(columns="index_right")

# %%
# dataframe of domestic EPC points not in a building footprints polygon
# these are definitely domestic properties but aren't being included in final planning tool because of this
points_join = gpd.sjoin(
    domestic_uprns_gdf, building_footprints_gdf, how="left", predicate="intersects"
)

epc_points_out = (points_join[points_join["index_right"].isna()]).drop(
    "index_right", axis=1
)

# %%
# fraction of points in domestic epc data that aren't being joined to a building footprint
epc_out_frac = len(epc_points_out) / len(domestic_uprns_gdf) * 100
print(f"proportion of EPC points not joined to a building: {epc_out_frac:.2f}%")

# %%
# save building footprint geometry in separate column
building_footprints_gdf["saved_geom"] = building_footprints_gdf.geometry

# find nearest building to each domestic property that hasn't been assigned a footprint
nearest_building = epc_points_out.sjoin_nearest(building_footprints_gdf, how="left")
# save this distance in a column
nearest_building["distance"] = nearest_building["geometry"].distance(
    nearest_building["saved_geom"]
)

# %%
# fraction of domestic buildings not assigned a footprint that are within X metres of a building
dist = 1
dist_fraction = (
    len(nearest_building[nearest_building["distance"] < dist])
    / len(nearest_building)
    * 100
)
print(
    f"proportion of EPC points not joined to a building that are within {dist}m of a building: {dist_fraction:.2f}%"
)

# %%
# histogram of distance from nearest building
h = plt.hist(nearest_building["distance"], bins=200, density=True)
plt.xlabel("Distance from nearest building [m]")
plt.ylabel("Fraction of total counts")


# %%
import branca.colormap as cm


def plot_nearest_buildings(nearest_building: gpd.GeoDataFrame):
    """
    Plot the resulting polygons together for each tech type in Folium.

    Args:
        gdf (gpd.GeoDataFrame): dataframe with polygons to dissolve and plot and tech labels
        colours (dict): tech type labels and their corresponding colours for plotting

    Returns Folium map
    """

    # Convert dissolved tech polygons and buildings to EPSG 4326 for plotting
    nearest_building_4326_gdf = nearest_building.to_crs(epsg=4326)
    nearest_building_4326_gdf = nearest_building_4326_gdf.set_geometry("saved_geom")
    nearest_building_4326_gdf = nearest_building_4326_gdf.to_crs(epsg=4326)

    # Get centre of boundary to centre map
    boundary_4326 = nearest_building_4326_gdf["geometry"].values[0]
    centre_map = shapely.get_coordinates(boundary_4326.centroid)

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    # Plot building footprints of nearest buildings
    for _, r in nearest_building_4326_gdf.iterrows():
        sim_geo = gpd.GeoSeries(r["saved_geom"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x: {
                "color": "blue",
                "weight": 0.1,
                "fillOpacity": 0.4,
            },
        )
        geo_j.add_to(m)

    # Plot UPRNS of missed EPC data
    for _, r in nearest_building_4326_gdf.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        min = np.min(nearest_building_4326_gdf["distance"])
        max = np.max(nearest_building["distance"])
        colours = cm.linear.YlOrRd_03.scale(min, max)
        marker = folium.Circle(
            radius=5,
            fill_color=colours(r["distance"]),
            fill_opacity=0.5,
            color="black",
            weight=1,
        )
        geo_j = folium.GeoJson(data=geo_j, marker=marker)
        geo_j.add_to(m)

    return m


# %%
# plot UPRNS that are within 1m of a building, with building footprints in blue
plot_nearest_buildings(nearest_building)

# %% [markdown]
# ## Properties labelled domestic but should be commercial

# %%
COLOURS = {
    "Individual solution": "#18A48C",
    "District heat network": "#EA2541",
    "Individual solution or Networked GSHP": "grey",
    "Networked GSHP": "#0000FF",
    "Communal solutions": "#FF6E47",
    "Individual solution or District heat network": "gray",
}


# %%
def plot_buildings(buildings_gdf: gpd.GeoDataFrame, colours: dict = COLOURS):
    """
    Plot the resulting polygons together for each tech type in Folium.

    Args:
        gdf (gpd.GeoDataFrame): dataframe with polygons to dissolve and plot and tech labels
        colours (dict): tech type labels and their corresponding colours for plotting

    Returns Folium map
    """
    # Dissolve by tech type
    dissolved_gdf = buildings_gdf.dissolve(
        by="1st_most_suitable_solution"
    ).reset_index()
    dissolved_gdf["colour"] = dissolved_gdf["1st_most_suitable_solution"].map(colours)

    # Convert dissolved tech polygons and buildings to EPSG 4326 for plotting
    dissolved_4326_gdf = dissolved_gdf.to_crs(epsg=4326)

    # Get centre of boundary to centre map
    boundary_4326 = dissolved_4326_gdf["geometry"].values[0]
    centre_map = shapely.get_coordinates(boundary_4326.centroid)

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    # Plot building footprints
    for _, r in dissolved_4326_gdf.iterrows():
        colour = colours[r["1st_most_suitable_solution"]]
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x, colour=colour: {
                "fillColor": colour,
                "fillOpacity": 0.3,
            },
        )
        folium.Popup(f"UPRNs: {r['UPRN']}").add_to(geo_j)
        geo_j.add_to(m)

    return m


# %%
# add column with area per UPRN for each building

tech_types_gdf["area_per_UPRN"] = (tech_types_gdf.area) / tech_types_gdf["UPRN"]
np.average(tech_types_gdf["area_per_UPRN"])  # for reference

# %%
# buildings with area per UPRN > 1000 m2
plot_buildings(tech_types_gdf[tech_types_gdf["area_per_UPRN"] > 1000])

# %%
# join tech types gdf with epc to get buildings with a confirmed domestic property
tech_with_epc = domestic_uprns_gdf.sjoin(
    tech_types_gdf, how="inner", predicate="intersects"
)

# how many of these have an area per UPRN > 1000 ?
tech_with_epc[tech_with_epc["area_per_UPRN"] > 1000]

# %%
# join tech types gdf with important buildings
tech_with_important = tech_types_gdf.sjoin(
    important_buildings_gdf, how="inner", predicate="intersects"
)

# important buildings that have just one UPRN. These should then be purely commercial buildings but are getting classed as domestic
tech_with_important[tech_with_important["UPRN"] == 1]

# %%
# same as above, with POI data

tech_with_poi = tech_types_gdf.sjoin(poi_gdf, how="inner", predicate="intersects")
tech_with_poi[tech_with_poi["UPRN"] == 1]
