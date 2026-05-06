# %% [markdown]
# ## Compare council tax data to pipeline domestic UPRNs

# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
import matplotlib.pyplot as plt
import shapely
import folium
import os
from sklearn.metrics import roc_auc_score, roc_curve

# %%
# local imports
from asf_heat_pump_suitability import PROJECT_DIR
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import (
    load_geodata,
    load_boundaries,
    base_getters,
)
from asf_heat_pump_suitability.pipeline.transform import uprns

# %%
council_tax_df = pd.read_csv(config["data"]["geodata"]["council_tax_data"]["plymouth"])
pipeline_domestic_gdf = base_getters.load_df_from_s3(
    config["data"]["processed"]["plymouth_residential_uprns"]
)
building_footprints_gdf = load_geodata.load_gdf_os_openmap_layer(
    layer="building", grid_squares="SX"
)
la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
    select_las="Plymouth"
)
uprns_df = load_geodata.load_df_osopen_uprn()
uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

# %%
building_footprints_gdf = building_footprints_gdf.sjoin(
    la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
    how="inner",
    predicate="intersects",
).drop(columns="index_right")

# %%
council_tax_df

# %%
# total number of council tax records (before empty UPRN/ coordinate rows are removed)
n_council_tax_records = len(council_tax_df)

# remove empty UPRN rows
council_tax_df = council_tax_df[council_tax_df["UPRN"] != ""]
# remove empty coordinate rows
council_tax_df = council_tax_df[council_tax_df["EASTING"] != ""]
council_tax_df = council_tax_df[council_tax_df["NORTHING"] != ""]

# %%
# convert council tax data to gdf
council_tax_gdf = gpd.GeoDataFrame(
    council_tax_df,
    geometry=gpd.points_from_xy(council_tax_df["EASTING"], council_tax_df["NORTHING"]),
    crs="EPSG:27700",
).drop_duplicates("UPRN")

# convert pipeline UPRN data to gdf
pipeline_domestic_gdf = uprns.generate_gdf_uprn_coords(pipeline_domestic_gdf)
pipeline_domestic_gdf = pipeline_domestic_gdf.drop_duplicates("UPRN")

# %% [markdown]
# ## How many UPRNs are in each dataset

# %%
# total number of council tax records (including empty UPRN/ coordinate rows)
print(f"total number of records in council tax data: {n_council_tax_records}")

# %%
# number of UPRNs with a coordinate in council tax data
print(
    f"number of unique URPNs in council tax data: {(council_tax_gdf['UPRN']).nunique()}"
)

# %%
# number of UPRNs identified as domestic in pipeline data
print(
    f"number of unique URPNs in pipeline data: {(pipeline_domestic_gdf['UPRN']).nunique()}"
)

# %% [markdown]
# ## Council tax UPRNs not identified by pipeline

# %%
council_tax_gdf["UPRN"] = council_tax_gdf["UPRN"].dropna().astype("int64")

# %%
# add column to council tax data with True/ False flag for if the UPRN is contained in the pipeline UPRNs
council_tax_gdf["in_pipeline"] = council_tax_gdf["UPRN"].isin(
    pipeline_domestic_gdf["UPRN"].tolist()
)

# %%
# how many council tax UPRNs are being missed by the pipeline?
print(
    f"number of domestic UPRNs in council tax data not picked up by pipeline: {sum(~council_tax_gdf['in_pipeline'])}"
)
# what is this as a percentage?
print(
    f"proportion of domestic UPRNs in council tax data not picked up by pipeline: {(sum(~council_tax_gdf['in_pipeline']))*100/ len(council_tax_gdf):.2f}%"
)

# %% [markdown]
# ## Pipeline domestic UPRNs not in council tax data
# - Most of these are probably not actually domestic.
# - There are some council tax records without a matching UPRN, so it's possible that some UPRNs in the pipeline are truly domestic but aren't being verified here.
# - For the same reason, there could be more true domestic UPRNs not being picked up by the pipeline that aren't being identified here.

# %%
# add a True/ False flag to the pipeline UPRNs if the UPRN is in the council tax data
pipeline_domestic_gdf["in_council_tax"] = pipeline_domestic_gdf["UPRN"].isin(
    council_tax_gdf["UPRN"].tolist()
)

# %%
pipeline_domestic_gdf

# %%
# how many UPRNs in the pipeline are not found in the council tax records?
print(
    f"number of UPRNs identified as domestic in pipeline not in council tax data: {sum(~pipeline_domestic_gdf['in_council_tax'])}"
)
# what is this as a percentage?
print(
    f"proportion of UPRNs identified as domestic in pipeline not in council tax data: {(sum(~pipeline_domestic_gdf['in_council_tax'])*100)/ len(pipeline_domestic_gdf):.2f}%"
)

# %% [markdown]
# ## Pipeline domestic building footprints with no council tax UPRNs in them
#
# These are fully non-domestic buildings and we'd like to remove as many of them from the pipeline as possible

# %%
# join pipeline UPRNs to building footprint data
pipeline_with_buildings = building_footprints_gdf.sjoin(
    pipeline_domestic_gdf, how="inner", predicate="contains"
).drop("index_right", axis=1)

# %%
# group by building to get aggrgated features
# at_least_1_domestic = building contains at least 1 confirmed domestic UPRN
# at_least_1_not_domestic = building contains at least 1 UPRN not in council tax data (so likely non domestic)
# pipeline_UPRN_count = number of pipeline UPRNs identified for that building
pipeline_with_buildings = (
    pipeline_with_buildings.groupby("ID")
    .agg(
        geometry=("geometry", "first"),
        at_least_1_domestic=("in_council_tax", "any"),
        at_least_1_not_domestic=("in_council_tax", lambda x: ~x.all()),
        pipeline_UPRN_count=("UPRN", "count"),
    )
    .reset_index()
)
pipeline_with_buildings = gpd.GeoDataFrame(
    pipeline_with_buildings,
    geometry=pipeline_with_buildings["geometry"],
    crs=config["constant"]["target_crs"],
)

# %%
# how many pipeline building footprints have at least 1 UPRN in the council tax data?
print(
    f"proportion of pipeline building footprint with at least 1 council tax UPRN: {100*sum(pipeline_with_buildings['at_least_1_domestic'])/len(pipeline_with_buildings):.2f}%"
)

# %%
# how many pipeline building footprints have at least 1 UPRN that is not found in the council tax data?
print(
    f"proportion of pipeline building footprints with at least 1 UPRN not in council tax data: {100*sum(pipeline_with_buildings['at_least_1_not_domestic'])/len(pipeline_with_buildings):.2f}%"
)

# %%
# how many pipeline building footprints have no UPRNs found in the council tax data? These are buildings we want to remove from the pipeline
print(
    f"Number of building footprints with no council tax UPRNs in them: {sum(~pipeline_with_buildings['at_least_1_domestic'])}"
)

# %%
print(
    f"Proportion of building footprints with no council tax UPRNs in them: {sum(~pipeline_with_buildings['at_least_1_domestic'])*100/len(pipeline_with_buildings):.2f}%"
)


# %%
def plot_buildings(
    pipeline_buildings_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame
):
    """
    Plot the pipeline building footprint polygons in a given boundary

    Args:
        pipeline_buildings_gdf (gpd.GeoDataFrame): dataframe of buildings assigned as residential, joined to building footprint polygons. Must contain a UPRN count column.
        boundary_gdf (gpd.GeoDataFrame): dataframe with LA boundaries

    Returns Folium map
    """
    pipeline_buildings_gdf_4326 = pipeline_buildings_gdf.to_crs(epsg=4326)
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
    for _, r in pipeline_buildings_gdf_4326.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x, colour="blue": {
                "fillColor": colour,
                "fillOpacity": 0.3,
            },
        )
        folium.Popup(f"UPRNs: {r['pipeline_UPRN_count']}").add_to(geo_j)
        geo_j.add_to(m)

    return m


# %%
# plot buildings identified by the pipeline as being domestic, but have no council tax UPRNs in them
# these are the buildings we want to remove from the pipeline
plot_buildings(
    pipeline_with_buildings[~pipeline_with_buildings["at_least_1_domestic"]],
    la_boundaries_gdf,
)

# %% [markdown]
# ## Mixed- use buildings

# %%
# join council tax data to building footprints. These are all buildings with at least one comfirmed domestic property
"""
council_with_buildings = building_footprints_gdf.sjoin(
    council_tax_gdf, how="inner", predicate="contains"
).drop("index_right", axis=1)
"""

# %%
# joining council tax points to nearest building < 3m away
# for some reason using the max_distance parameter causes an error here so using this manual workaround for now
council_with_buildings = council_tax_gdf.sjoin_nearest(
    building_footprints_gdf, how="inner", distance_col="dist"
)
council_with_buildings = council_with_buildings[council_with_buildings["dist"] <= 3]
council_with_buildings = council_with_buildings.drop(["index_right", "dist"], axis=1)

# %%
council_with_buildings

# %%
# aggregate features by building ID
# council_UPRN_count = number of UPRNs in that building that are in the council tax data
# in_pipeline = the building is present in the pipeline domestic buildings
council_with_buildings = (
    council_with_buildings.groupby("ID")
    .agg(
        geometry=("geometry", "first"),
        council_UPRN_count=("UPRN", "count"),
        in_pipeline=("in_pipeline", "any"),
    )
    .reset_index()
)
council_with_buildings = gpd.GeoDataFrame(
    council_with_buildings,
    geometry=council_with_buildings["geometry"],
    crs=config["constant"]["target_crs"],
)

# %%
council_with_buildings

# %%
# join all OS UPRNs to building footprints
all_uprns_with_buildings = building_footprints_gdf.sjoin(
    uprns_gdf, how="inner", predicate="contains"
).drop("index_right", axis=1)

# %%
# aggregate features by building ID
# total_UPRN_count = total number of OS UPRNs in that building
all_uprns_with_buildings = (
    all_uprns_with_buildings.groupby("ID")
    .agg(geometry=("geometry", "first"), total_UPRN_count=("UPRN", "count"))
    .reset_index()
)
all_uprns_with_buildings = gpd.GeoDataFrame(
    all_uprns_with_buildings,
    geometry=all_uprns_with_buildings["geometry"],
    crs=config["constant"]["target_crs"],
)

# %%
# merge with council tax buildings
# find buildings where the number of council tax UPRNs != total number of UPRNs. These are mixed-use buildings
mixed_use_gdf = council_with_buildings.merge(
    all_uprns_with_buildings.drop("geometry", axis=1), how="left", on="ID"
)
mixed_use_gdf = mixed_use_gdf[
    mixed_use_gdf["council_UPRN_count"] != mixed_use_gdf["total_UPRN_count"]
]

# %%
print(
    f"proportion of all building footprints which are mixed use: {100*len(mixed_use_gdf)/len(all_uprns_with_buildings):.2f}%"
)

# %% [markdown]
# ## How many of these mixed-use buildings are in the pipeline?
#
# Here, I am looking at buildings we want to keep as they have at least one confirmed domestic property. They however are actually mixed-use buildings, so will contain some non-domestic properties (or properties missing from the council tax data for some reason).

# %%
# merge confirmed mixed-use buildings with pipeline buildings to find buildings identified by the pipeline that are actually mixed-use
pipeline_mixed_use_gdf = pipeline_with_buildings.merge(
    mixed_use_gdf.drop("geometry", axis=1), how="inner", on="ID"
)

# %%
pipeline_mixed_use_gdf

# %%
print(
    f"proportion of pipeline building footprints which are mixed use: {100*len(pipeline_mixed_use_gdf)/len(pipeline_with_buildings):.2f}%"
)


# %%
def plot_pipeline_mixed_buildings(
    pipeline_buildings_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame
):
    """
    Plot the building footprint polygons in a given boundary

    Args:
        pipeline_buildings_gdf (gpd.GeoDataFrame): dataframe of buildings assigned as residential, joined to building footprint polygons. Must contain a UPRN count column.
        boundary_gdf (gpd.GeoDataFrame): dataframe with LA boundaries

    Returns Folium map
    """
    pipeline_buildings_gdf_4326 = pipeline_buildings_gdf.to_crs(epsg=4326)
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
    for _, r in pipeline_buildings_gdf_4326.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x, colour="blue": {
                "fillColor": colour,
                "fillOpacity": 0.3,
            },
        )
        folium.Popup(
            f"pipeline UPRNs: {r['pipeline_UPRN_count']} <br> domestic UPRNs: {r['council_UPRN_count']} <br> total UPRNs: {r['total_UPRN_count']}",
            max_width=200,
        ).add_to(geo_j)
        geo_j.add_to(m)

    return m


# %%
# mixed use buildings in the pipeline
plot_pipeline_mixed_buildings(pipeline_mixed_use_gdf, la_boundaries_gdf)

# %% [markdown]
# ## Where are we identifying the wrong number of UPRNs?
# Not including cases where the whole building is misidentified

# %%
wrong_uprn_number = pipeline_with_buildings.merge(
    council_with_buildings.drop("geometry", axis=1), how="inner", on="ID"
)

# %%
# find buildings where the number of UPRNs identified by the pipeline is different to the number of UPRNs in the council tax data
# create a column 'UPRN_difference' to store this difference
wrong_uprn_number = wrong_uprn_number[
    wrong_uprn_number["pipeline_UPRN_count"] != wrong_uprn_number["council_UPRN_count"]
]
wrong_uprn_number["UPRN_difference"] = (
    wrong_uprn_number["pipeline_UPRN_count"] - wrong_uprn_number["council_UPRN_count"]
)

# %%
# buildings where pipeline is identifying fewer UPRNs that council tax (not including cases where whole building is missing)
(
    -1
    * wrong_uprn_number[
        wrong_uprn_number["pipeline_UPRN_count"]
        < wrong_uprn_number["council_UPRN_count"]
    ]["UPRN_difference"]
).describe()

# %%
# buildings where pipeline is identifying more UPRNs that council tax (not including cases where whole building is wrong)
wrong_uprn_number[
    wrong_uprn_number["pipeline_UPRN_count"] > wrong_uprn_number["council_UPRN_count"]
]["UPRN_difference"].describe()


# %%
def plot_wrong_uprn_count(
    buildings_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame
):
    """
    Plot the building footprint polygons where we are identiifying the wrong number of UPRNs (not including cases where the whole building is wrong)

    Args:
        residential_buildings_gdf (gpd.GeoDataFrame): dataframe of buildings assigned as residential, joined to building footprint polygons
        boundary_gdf (gpd.GeoDataFrame): dataframe with LA boundaries
        area (float): area per UPRN above which you want to plot

    Returns Folium map
    """
    buildings_gdf_4326 = buildings_gdf.to_crs(epsg=4326)
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
    for _, r in buildings_gdf_4326.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        if r["pipeline_UPRN_count"] > r["council_UPRN_count"]:
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="blue": {
                    "fillColor": colour,
                    "weight": 0.1,
                    "fillOpacity": 0.3,
                },
            )

        if r["pipeline_UPRN_count"] < r["council_UPRN_count"]:
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="red": {
                    "fillColor": colour,
                    "weight": 0.1,
                    "fillOpacity": 0.3,
                },
            )
        folium.Popup(
            f"pipeline UPRNs minus council tax UPRNs: {r['UPRN_difference']}"
        ).add_to(geo_j)
        geo_j.add_to(m)
    return m


# %%
# blue = we are identifying more UPRNs than in council tax data
# red = we are identifying fewer UPRNs than in council tax data
plot_wrong_uprn_count(wrong_uprn_number, la_boundaries_gdf)

# %%
import branca.colormap as cm


def plot_heat_map_wrong_uprn_count(
    buildings_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame
):
    """
    Plot the building footprint polygons where we are identiifying the wrong number of UPRNs (not including cases where the whole building is wrong)

    Args:
        residential_buildings_gdf (gpd.GeoDataFrame): dataframe of buildings assigned as residential, joined to building footprint polygons
        boundary_gdf (gpd.GeoDataFrame): dataframe with LA boundaries
        area (float): area per UPRN above which you want to plot

    Returns Folium map
    """
    buildings_gdf_4326 = buildings_gdf.to_crs(epsg=4326)
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
    min = np.min(buildings_gdf_4326["UPRN_difference"])
    max = np.max(buildings_gdf_4326["UPRN_difference"])
    colours = cm.linear.YlOrRd_03.scale(min, max)
    for _, r in buildings_gdf_4326.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j,
            style_function=lambda x, colour=colours(r["UPRN_difference"]): {
                "fillColor": colour,
                "weight": 0.1,
                "fillOpacity": 0.7,
            },
        )
        folium.Popup(
            f"pipeline UPRNs minus council tax UPRNs: {r['UPRN_difference']}"
        ).add_to(geo_j)
        geo_j.add_to(m)
    return m


# %%
# heat map of UPRN difference
plot_heat_map_wrong_uprn_count(wrong_uprn_number, la_boundaries_gdf)

# %% [markdown]
# ## Identify features to classify as domestic / non-domestic

# %%
# label all plymouth buildings
council_with_buildings

# %%
all_uprns_with_buildings

# %%
# create a column 'domestic' with True/ False if the building is in the list of buildings that have a council tax UPRN in them
all_uprns_with_buildings["domestic"] = all_uprns_with_buildings["ID"].isin(
    council_with_buildings["ID"].to_list()
)

# %%
all_uprns_with_buildings

# %%
# create features
all_uprns_with_buildings["area_m2"] = all_uprns_with_buildings.area

# %%
all_uprns_with_buildings

# %%
all_uprns_with_buildings["area_per_UPRN"] = (
    all_uprns_with_buildings["area_m2"] / all_uprns_with_buildings["total_UPRN_count"]
)

# %%
all_uprns_with_buildings

# %%
# summary statistics for each area per UPRN:
# buildings with at least 1 domestic property
all_uprns_with_buildings[all_uprns_with_buildings["domestic"]][
    "area_per_UPRN"
].describe()

# %%
# buildings with no domestic properties
all_uprns_with_buildings[~(all_uprns_with_buildings["domestic"])][
    "area_per_UPRN"
].describe()

# %%
# are histograms different
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].hist(
    all_uprns_with_buildings[all_uprns_with_buildings["domestic"]]["area_per_UPRN"],
    bins=100,
)
axs[0].set_title("Domestic")
axs[1].hist(
    all_uprns_with_buildings[~(all_uprns_with_buildings["domestic"])]["area_per_UPRN"],
    bins=100,
)
axs[1].set_title("Non-domestic")
axs[1].set_xlabel("Area per UPRN [m$^2$]")

# %%
# large outlier in non-domestic making this plot hard to see, zoom in to first 5000 m^2
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].hist(
    all_uprns_with_buildings[all_uprns_with_buildings["domestic"]]["area_per_UPRN"],
    bins=100,
    range=(0, 5000),
)
axs[0].set_title("Domestic")
axs[1].hist(
    all_uprns_with_buildings[~(all_uprns_with_buildings["domestic"])]["area_per_UPRN"],
    bins=100,
    range=(0, 5000),
)
axs[1].set_title("Non-domestic")
axs[1].set_xlabel("Area per UPRN [m$^2$]")

# %%
auc_area_per_UPRN = roc_auc_score(
    ~all_uprns_with_buildings["domestic"], all_uprns_with_buildings["area_per_UPRN"]
)

print(f"ROC AUC Score for area per UPRN: {auc_area_per_UPRN:.4f}")

# %%
auc_area = roc_auc_score(
    ~all_uprns_with_buildings["domestic"], all_uprns_with_buildings["area_m2"]
)

print(f"ROC AUC Score for building foorprint area: {auc_area:.4f}")

# %%
# Youden's J-score

fpr, tpr, thresholds = roc_curve(
    ~all_uprns_with_buildings["domestic"], all_uprns_with_buildings["area_per_UPRN"]
)

# Calculate J for every threshold
j_scores = tpr - fpr

# Find the index of the highest J score
index = np.argmax(j_scores)
best_threshold = thresholds[index]

print(f"Best value of area per UPRN to use: {best_threshold:.2f}")
print(f"Maximum Youden's Index (J) is: {j_scores[index]:.4f}")

# %%
# get 95% of the domestic buildings
target = 0.95
target_fpr = 1 - target

# Find the threshold area per UPRN that gives us our target FPR
index = np.argmin(np.abs(fpr - target_fpr))
threshold_95 = thresholds[index]

print(f"To retain 95% of domestic buildings, use area per UPRN: {threshold_95:.2f}")
print(f"This removed {tpr[index]*100:.2f}% of non-domestic buildings.")


# %%
def plot_building_feature(
    buildings_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame
):
    """
    Plot the building footprint polygons with area per UPRN > area specified in function

    Args:
        buildings_gdf (gpd.GeoDataFrame): dataframe of building footprint polygons
        boundary_gdf (gpd.GeoDataFrame): dataframe with LA boundaries
        area (float): area per UPRN above which you want to plot

    Returns Folium map
    """
    buildings_gdf_4326 = buildings_gdf.to_crs(epsg=4326)
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
    for _, r in buildings_gdf_4326.iterrows():
        sim_geo = gpd.GeoSeries(r["geometry"])
        geo_j = sim_geo.to_json()
        if r["domestic"]:
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="blue": {
                    "fillColor": colour,
                    "weight": 0.1,
                    "fillOpacity": 0.3,
                },
            )
        else:
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="red": {
                    "fillColor": colour,
                    "weight": 0.1,
                    "fillOpacity": 0.3,
                },
            )
        geo_j.add_to(m)

    return m


# %%
# blue = domestic
# red = non-domestic
plot_building_feature(
    all_uprns_with_buildings[all_uprns_with_buildings["area_per_UPRN"] > threshold_95],
    la_boundaries_gdf,
)

# %%
# how many additional buildings that we are not already removing would this threshold remove?
pipeline_with_buildings["area_per_UPRN"] = (
    pipeline_with_buildings.area / pipeline_with_buildings["pipeline_UPRN_count"]
)
sum(
    ~pipeline_with_buildings[pipeline_with_buildings["area_per_UPRN"] > best_threshold][
        "at_least_1_domestic"
    ]
)

# %%
# how many domestic buildings would this remove?
# This number is quite high which adds to the idea that MCC looks to be a better method for determining this threshold that Youden's J
sum(
    pipeline_with_buildings[pipeline_with_buildings["area_per_UPRN"] > best_threshold][
        "at_least_1_domestic"
    ]
)

# %% [markdown]
# ## Council tax UPRNs not joined to a building footprint

# %%
# re-doing the join here
council_with_buildings = council_tax_gdf.sjoin(
    building_footprints_gdf, how="left", predicate="intersects"
).drop("index_right", axis=1)

# %%
# these are council tax UPRNs that have not been joined to a building footprint (so building ID is NaN)
council_outside_buildings = council_with_buildings[council_with_buildings["ID"].isna()]

# %%
print(
    f"proportion of council tax UPRNs not joined to a building footprint: {100*len(council_outside_buildings)/len(council_tax_gdf):.2f}%"
)

# %%
# for these UPRNs, find the nearest building footprint
council_outside_buildings = council_outside_buildings.sjoin_nearest(
    building_footprints_gdf, how="inner", distance_col="distance"
)

# %%
# what is the distribution of distance to nearest footprint for these UPRNs
council_outside_buildings["distance"].describe()

# %%
# how many are with 1m?
distance = 1
print(
    f"proportion of council tax UPRNs not joined to a building footprint within {distance}m of a building: {100*len(council_outside_buildings[council_outside_buildings['distance']<distance])/(len(council_outside_buildings)):.2f}%"
)
