# %% [markdown]
# ## Domestic Filtering Exploration
#
# Notebook to explore properties labelled as domestic/ commercial to identify those that have been incorrectly classified
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
    load_geodata,
    load_boundaries,
    base_getters,
)
from asf_heat_pump_suitability.pipeline.transform import uprns, poi

# %%
# load datasets
building_footprints_gdf = load_geodata.load_gdf_os_openmap_layer(
    layer="building", grid_squares="SX"
)

tech_types_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_building_most_suitable_tech.geojson"
).to_crs(27700)
important_buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
    layer="important_building", grid_squares="SX"
)
epc_domestic_set = uprns.load_set_valid_epc_uprns(epc_type="domestic")
epc_commercial_set = uprns.load_set_valid_epc_uprns(epc_type="commercial")
uprns_df = load_geodata.load_df_osopen_uprn()
uprns_GB_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

# %%
residential_uprn_df = base_getters.load_df_from_s3(
    config["data"]["processed"]["plymouth_residential_uprns"]
)
residential_gdf = uprns.generate_gdf_uprn_coords(residential_uprn_df)
residential_gdf = building_footprints_gdf.sjoin(
    residential_gdf, how="inner", predicate="intersects"
).drop(columns="index_right")

# %%
# Load POI data

poi_gdf = load_geodata.load_gdf_poi()
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
points_join = domestic_uprns_gdf.sjoin(
    building_footprints_gdf, how="left", predicate="intersects"
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
    f"proportion of EPC points in Plymouth not joined to a building that are within {dist}m of a building: {dist_fraction:.2f}%"
)

# %%
nearest_building["distance"].describe()

# %%
# histogram of distance from nearest building
h = plt.hist(nearest_building["distance"], bins=200, density=False)
plt.xlabel("Distance from nearest building [m]")
plt.ylabel("Total counts")

# %%
import branca.colormap as cm


def plot_nearest_buildings(
    nearest_building: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame
):
    """
    Plot the points from EPC register that were not joined to a building footprint, along with their nearest building footprint.

    Args:
        nearest_building (gpd.GeoDataFrame): dataframe with EPC point geometries not joined to a building footprint, and the nearest building footprint polygon
        boundary_gdf (gpd.GeoDataFrame): local authority boundaries

    Returns Folium map
    """

    # Convert EPC points and building footprints to EPSG 4326 for plotting
    nearest_building_4326_gdf = nearest_building.to_crs(epsg=4326)
    nearest_building_4326_gdf = nearest_building_4326_gdf.set_geometry("saved_geom")
    nearest_building_4326_gdf = nearest_building_4326_gdf.to_crs(epsg=4326)

    # Get centre of boundary to centre map
    boundary_4326 = boundary_gdf.to_crs(epsg=4326)["geometry"].values[0]
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
    min = np.min(nearest_building_4326_gdf["distance"])
    max = np.max(nearest_building_4326_gdf["distance"])
    colours = cm.linear.YlOrRd_03.scale(min, max)
    for _, r in nearest_building_4326_gdf.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
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
# plot UPRNS not joined to a building, with building footprints in blue
plot_nearest_buildings(nearest_building, boundary_gdf=la_boundaries_gdf)

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
def plot_buildings(
    residential_buildings_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    area: float,
):
    """
    Plot the building footprint polygons with area per UPRN > area specified in function

    Args:
        residential_buildings_gdf (gpd.GeoDataFrame): dataframe of buildings assigned as residential, joined to building footprint polygons
        boundary_gdf (gpd.GeoDataFrame): dataframe with LA boundaries
        area (float): area per UPRN above which you want to plot

    Returns Folium map
    """
    residential_buildings_gdf_4326 = residential_buildings_gdf.to_crs(epsg=4326)
    # Get centre of boundary to centre map
    boundary_4326 = boundary_gdf.to_crs(epsg=4326)["geometry"].values[0]
    centre_map = shapely.get_coordinates(boundary_4326.centroid)

    # Create map
    m = folium.Map(
        location=[centre_map[0][1], centre_map[0][0]],
        zoom_start=15,
        tiles="esri_worldimagery",
    )

    # Plot building footprints
    for _, r in residential_buildings_gdf_4326[
        residential_buildings_gdf_4326["area_per_UPRN"] > area
    ].iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x, colour="blue": {
                "fillColor": colour,
                "fillOpacity": 0.3,
            },
        )
        folium.Popup(f"UPRNs: {r['UPRN_count']}").add_to(geo_j)
        folium.Popup(f"area per UPRN: {r['area_per_UPRN']}").add_to(geo_j)
        geo_j.add_to(m)

    return m


# %%
# 1. Create a 'count' column (using any existing column to store the count)
# We use 'geometry' to group, and count how many times that geometry appears
uprn_counts = residential_gdf.groupby("geometry").size().reset_index(name="UPRN_count")

# 2. Convert the result back into a GeoDataFrame
# (Groupby results often revert to a standard DataFrame)
residential_with_uprn_counts_gdf = gpd.GeoDataFrame(
    uprn_counts, geometry="geometry", crs=residential_gdf.crs
)

# %%
# add column with area per UPRN for each building
residential_with_uprn_counts_gdf["area_per_UPRN"] = (
    residential_with_uprn_counts_gdf.area
    / residential_with_uprn_counts_gdf["UPRN_count"]
)
np.average(residential_with_uprn_counts_gdf["area_per_UPRN"])  # for reference

# %%
(residential_with_uprn_counts_gdf["area_per_UPRN"]).describe()

# %%
# buildings with area per UPRN > 1000 m2
plot_buildings(
    residential_buildings_gdf=residential_with_uprn_counts_gdf,
    boundary_gdf=la_boundaries_gdf,
    area=1000,
)

# %%
# join tech types gdf with epc to get buildings with a confirmed domestic property
residential_with_epc = domestic_uprns_gdf.sjoin(
    residential_with_uprn_counts_gdf, how="inner", predicate="intersects"
)

# how many of these have an area per UPRN > 1000 ?
residential_with_epc[residential_with_epc["area_per_UPRN"] > 1000]

# %%
(residential_with_epc["area_per_UPRN"]).describe()

# %%
# histogram of area per UPRN confirmed domestic properties
h = plt.hist(residential_with_epc["area_per_UPRN"], bins=200, density=False)
plt.xlabel("area per UPRN EPC domestic buildings [m^2]")
plt.ylabel("Total counts")

# %%
# join tech types gdf with important buildings
residential_with_important = residential_with_uprn_counts_gdf.sjoin(
    important_buildings_gdf, how="inner", predicate="intersects"
)

# important buildings that have just one UPRN. These should then be purely commercial buildings but are getting classed as domestic
residential_with_important[residential_with_important["UPRN_count"] == 1]

# %%
# same as above, with POI data

residential_with_poi = residential_with_uprn_counts_gdf.sjoin(
    poi_gdf, how="inner", predicate="intersects"
)
residential_with_poi[residential_with_poi["UPRN_count"] == 1]
