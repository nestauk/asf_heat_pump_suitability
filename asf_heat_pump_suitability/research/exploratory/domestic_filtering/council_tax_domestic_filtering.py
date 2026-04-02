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
    load_tree_input,
    load_geodata,
    load_boundaries,
    base_getters,
)
from asf_heat_pump_suitability.pipeline.transform import uprns, poi
from asf_heat_pump_suitability.pipeline.transform import outdoor_space

# %%
council_tax_gdf = gpd.read_file(
    config["data"]["geodata"]["council_tax_data"]["plymouth"]
)
pipeline_domestic_gdf = base_getters.load_df_from_s3(
    config["data"]["processed"]["plymouth_residential_uprns"]
)
building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="building", grid_squares="SX"
)
la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
    select_las="Plymouth"
)

# %%
# remove empty UPRN rows
council_tax_gdf = council_tax_gdf[council_tax_gdf["UPRN"] != ""]
# remove empty coordinate rows
council_tax_gdf = council_tax_gdf[council_tax_gdf["EASTING"] != ""]
council_tax_gdf = council_tax_gdf[council_tax_gdf["NORTHING"] != ""]

# %%
council_tax_gdf = gpd.GeoDataFrame(
    council_tax_gdf,
    geometry=gpd.points_from_xy(
        council_tax_gdf["EASTING"], council_tax_gdf["NORTHING"]
    ),
    crs="EPSG:27700",
)
pipeline_domestic_gdf = uprns.generate_gdf_uprn_coords(pipeline_domestic_gdf)

# %% [markdown]
# ## How many UPRNs are in each dataset

# %%
council_tax_gdf

# %%
council_tax_gdf = council_tax_gdf[council_tax_gdf["UPRN"] != ""]

# %%
pipeline_domestic_gdf

# %%
print(
    f"number of unique URPNs in council tax data: {(council_tax_gdf['UPRN']).nunique()}"
)

# %%
print(
    f"number of unique URPNs in pipeline data: {(pipeline_domestic_gdf['UPRN']).nunique()}"
)

# %% [markdown]
# ## Council tax UPRNs not identified by pipeline

# %%
council_tax_gdf["UPRN"] = council_tax_gdf["UPRN"].astype("int64")

# %%
council_tax_gdf["UPRN"]

# %%
# council tax UPRNs also in pipeline domestic UPRNs
council_in_pipeline = council_tax_gdf[
    council_tax_gdf["UPRN"].isin(pipeline_domestic_gdf["UPRN"].tolist())
]

# %%
council_in_pipeline

# %%
print(
    f"number of domestic UPRNs in council tax data not picked up by pipeline: {len(council_tax_gdf) - len(council_in_pipeline)}"
)
print(
    f"proportion of domestic UPRNs in council tax data not picked up by pipeline: {(len(council_tax_gdf) - len(council_in_pipeline))*100/ len(council_tax_gdf):.2f}%"
)

# %% [markdown]
# ## Pipeline domestic UPRNs not in council tax data
# These are probably not actually domestic

# %%
pipeline_not_in_council = pipeline_domestic_gdf[
    ~pipeline_domestic_gdf["UPRN"].isin(council_tax_gdf["UPRN"].tolist())
]

# %%
pipeline_not_in_council

# %%
print(
    f"number of UPRNs identified as domestic in pipeline not in council tax data: {len(pipeline_not_in_council)}"
)
print(
    f"proportion of UPRNs identified as domestic in pipeline not in council tax data: {(len(pipeline_not_in_council)*100)/ len(pipeline_domestic_gdf):.2f}%"
)

# %% [markdown]
# ## Pipeline domestic building footprints with no council tax UPRNs in them
#
# Fully commercial units

# %%
council_with_buildings = building_footprints_gdf.sjoin(
    council_tax_gdf, how="inner", predicate="contains"
).drop("index_right", axis=1)

# %%
council_with_buildings

# %%
pipeline_with_buildings = building_footprints_gdf.sjoin(
    pipeline_domestic_gdf, how="inner", predicate="contains"
).drop("index_right", axis=1)

# %%
# list of building footprints we identify as domestic with at least 1 UPRN in council tax data
pipeline_buildings_in_council_tax = pipeline_with_buildings[
    pipeline_with_buildings["UPRN"].isin(council_tax_gdf["UPRN"].tolist())
]

# %%
print(
    f"proportion of buildings footprints with at least 1 council tax UPRN: {100*len(pipeline_buildings_in_council_tax.drop_duplicates('geometry'))/len(pipeline_with_buildings.drop_duplicates('geometry')):.2f}%"
)

# %%
# list of building footprints we identify as domestic with at least 1 UPRN not in council tax data
pipeline_buildings_not_in_council_tax = pipeline_with_buildings[
    ~(pipeline_with_buildings["UPRN"].isin(council_tax_gdf["UPRN"].tolist()))
]

# %%
# pipeline domestic buildings with no council tax UPRN in them
pipeline_buildings_no_uprn = pipeline_with_buildings[
    ~pipeline_with_buildings["geometry"].isin(
        pipeline_buildings_in_council_tax["geometry"]
    )
]

# %%
print(
    f"proportion of building footprints with at least 1 UPRN not in council tax data: {100*len(pipeline_buildings_not_in_council_tax.drop_duplicates('geometry'))/len(pipeline_with_buildings.drop_duplicates('geometry')):.2f}%"
)

# %%
pipeline_buildings_no_uprn = pipeline_buildings_no_uprn.drop_duplicates("geometry")

# %%
uprn_counts = (
    pipeline_buildings_no_uprn.groupby("geometry").size().reset_index(name="UPRN_count")
)
pipeline_buildings_no_uprn = gpd.GeoDataFrame(
    uprn_counts, geometry="geometry", crs="EPSG:27700"
)

# %%
print(
    f"Number of building footprints with no council tax UPRNs in them: {len(uprn_counts)}"
)

# %%
print(
    f"Proportion of building footprints with no council tax UPRNs in them: {len(uprn_counts)*100/len(pipeline_with_buildings.drop_duplicates('geometry')):.2f}%"
)


# %%
def plot_buildings(
    pipeline_buildings_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame
):
    """
    Plot the building footprint polygons with area per UPRN > area specified in function

    Args:
        residential_buildings_gdf (gpd.GeoDataFrame): dataframe of buildings assigned as residential, joined to building footprint polygons
        boundary_gdf (gpd.GeoDataFrame): dataframe with LA boundaries
        area (float): area per UPRN above which you want to plot

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
        folium.Popup(f"UPRNs: {r['UPRN_count']}").add_to(geo_j)
        geo_j.add_to(m)

    return m


# %%
# plot buildings identified by the pipeline as being domestic, but have no council tax UPRNs in them
plot_buildings(pipeline_buildings_no_uprn, la_boundaries_gdf)

# %% [markdown]
# ## Mixed-use buildings

# %%
# buildings with at least 1 UPRN in council tax data (domestic) and at least 1 UPRN not in council tax data (commercial)
mixed_use_gdf = pipeline_buildings_not_in_council_tax.sjoin(
    pipeline_buildings_in_council_tax, how="inner", predicate="intersects"
)

# %%
mixed_use_gdf.drop_duplicates("geometry")

# %%
print(
    f"proportion of building footprints which are mixed use: {100*len(mixed_use_gdf.drop_duplicates('geometry'))/len(pipeline_with_buildings.drop_duplicates('geometry')):.2f}%"
)

# %%
uprn_counts = (
    pipeline_buildings_in_council_tax.groupby("geometry")
    .size()
    .reset_index(name="UPRN_count")
)
pipeline_buildings_in_council_tax = gpd.GeoDataFrame(
    uprn_counts, geometry="geometry", crs="EPSG:27700"
)

# %%
uprn_counts = (
    pipeline_buildings_not_in_council_tax.groupby("geometry")
    .size()
    .reset_index(name="UPRN_count")
)
pipeline_buildings_not_in_council_tax = gpd.GeoDataFrame(
    uprn_counts, geometry="geometry", crs="EPSG:27700"
)

# %%
pipeline_buildings_in_council_tax

# %%
pipeline_buildings_not_in_council_tax

# %%
mixed_use_with_uprn_counts = pipeline_buildings_in_council_tax.sjoin(
    pipeline_buildings_not_in_council_tax, how="inner", predicate="intersects"
).drop("index_right", axis=1)

# %%
mixed_use_with_uprn_counts.drop_duplicates("geometry")

# %%
mixed_use_with_uprn_counts = mixed_use_with_uprn_counts.rename(
    columns={
        "UPRN_count_left": "pipeline_UPRN_count",
        "UPRN_count_right": "council_tax_UPRN_count",
    }
)

# %%
# proportion of UPRNs in each pipeline building that are in council tax data (so are confirmed domestic)
mixed_use_with_uprn_counts["proportion_domestic_UPRN"] = (
    mixed_use_with_uprn_counts["council_tax_UPRN_count"]
    / mixed_use_with_uprn_counts["pipeline_UPRN_count"]
)

# %%
mixed_use_with_uprn_counts

# %%
mixed_use_with_uprn_counts[
    mixed_use_with_uprn_counts["pipeline_UPRN_count"]
    == mixed_use_with_uprn_counts["council_tax_UPRN_count"]
]
# why would we have some with the same number ???? unless the UPRNs are wrong ? or something here is wrong


# %%
def plot_mixed_use_buildings(
    buildings_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame
):
    """
    Plot the building footprint polygons with area per UPRN > area specified in function

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
        if r["pipeline_UPRN_count"] > r["council_tax_UPRN_count"]:
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="blue": {
                    "fillColor": colour,
                    "weight": 0.1,
                    "fillOpacity": 0.3,
                },
            )
        if r["pipeline_UPRN_count"] < r["council_tax_UPRN_count"]:
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="red": {
                    "fillColor": colour,
                    "weight": 0.1,
                    "fillOpacity": 0.3,
                },
            )
        if r["pipeline_UPRN_count"] == r["council_tax_UPRN_count"]:
            geo_j = folium.GeoJson(
                data=geo_j,
                style_function=lambda x, colour="orange": {
                    "fillColor": colour,
                    "weight": 0.1,
                    "fillOpacity": 0.3,
                },
            )
        geo_j.add_to(m)

    return m


# %%
# mixed use buildings
# blue = we are identifying more UPRNs as domestic than are in council tax data
# red = we are identifying fewer UPRNs as domestic than are in council tax data
# orange = same number (not sure why this is happening)
plot_mixed_use_buildings(mixed_use_with_uprn_counts, la_boundaries_gdf)

# %%
h = plt.hist(
    mixed_use_with_uprn_counts["proportion_domestic_UPRN"], bins=100, range=(0, 5)
)

# %% [markdown]
# ## Identify features to classify as domestic / non-domestic

# %%
# label all plymouth buildings
council_with_buildings

# %%
building_footprints_gdf = building_footprints_gdf.sjoin(
    la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
    how="inner",
    predicate="intersects",
).drop(columns="index_right")

# %%
building_footprints_gdf["domestic"] = np.where(
    building_footprints_gdf["ID"].isin(council_with_buildings["ID"].tolist()),
    True,
    False,
)

# %%
building_footprints_gdf

# %%
# create features
building_footprints_gdf["area_m2"] = building_footprints_gdf.area

# %%
uprns_df = load_geodata.load_df_osopen_uprn()
uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

# %%
building_footprints_with_uprns = building_footprints_gdf.sjoin(
    uprns_gdf, how="inner", predicate="contains"
).drop("index_right", axis=1)

# %%
building_footprints_with_uprns

# %%
uprn_counts = (
    building_footprints_with_uprns.groupby("geometry")
    .size()
    .reset_index(name="UPRN_count")
)
uprn_counts = gpd.GeoDataFrame(uprn_counts, geometry="geometry", crs="EPSG:27700")

# %%
building_footprints_gdf = building_footprints_gdf.sjoin(
    uprn_counts, how="inner", predicate="intersects"
).drop("index_right", axis=1)

# %%
building_footprints_gdf

# %%
building_footprints_gdf["area_per_UPRN"] = (
    building_footprints_gdf["area_m2"] / building_footprints_gdf["UPRN_count"]
)

# %%
building_footprints_gdf

# %%
# summary statistics for each area per UPRN:
# buildings with at least 1 domestic property
building_footprints_gdf[building_footprints_gdf["domestic"]]["area_per_UPRN"].describe()

# %%
# buildings with no domestic properties
building_footprints_gdf[building_footprints_gdf["domestic"] == False][
    "area_per_UPRN"
].describe()

# %%
# are histograms different
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].hist(
    building_footprints_gdf[building_footprints_gdf["domestic"]]["area_per_UPRN"],
    bins=100,
)
axs[0].set_title("Domestic")
axs[1].hist(
    building_footprints_gdf[building_footprints_gdf["domestic"] == False][
        "area_per_UPRN"
    ],
    bins=100,
)
axs[1].set_title("Non-domestic")
axs[1].set_xlabel("Area per UPRN [m$^2$]")

# %%
# large outlier in non-domestic making this plot hard to see, zoom in to first 5000 m^2
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].hist(
    building_footprints_gdf[building_footprints_gdf["domestic"]]["area_per_UPRN"],
    bins=100,
    range=(0, 5000),
)
axs[0].set_title("Domestic")
axs[1].hist(
    building_footprints_gdf[building_footprints_gdf["domestic"] == False][
        "area_per_UPRN"
    ],
    bins=100,
    range=(0, 5000),
)
axs[1].set_title("Non-domestic")
axs[1].set_xlabel("Area per UPRN [m$^2$]")

# %%
auc_area_per_UPRN = roc_auc_score(
    ~building_footprints_gdf["domestic"], building_footprints_gdf["area_per_UPRN"]
)

print(f"ROC AUC Score for area per UPRN: {auc_area_per_UPRN:.4f}")

# %%
auc_area = roc_auc_score(
    ~building_footprints_gdf["domestic"], building_footprints_gdf["area_m2"]
)

print(f"ROC AUC Score for building foorprint area: {auc_area:.4f}")

# %%
# Youden's J-score

fpr, tpr, thresholds = roc_curve(
    ~building_footprints_gdf["domestic"], building_footprints_gdf["area_per_UPRN"]
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
    building_footprints_gdf[building_footprints_gdf["area_per_UPRN"] > threshold_95],
    la_boundaries_gdf,
)

# %%
