# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Merging the Plymouth UPRNs into one polygon per cluster
#
# This notebook creates polygons for each cluster of properties in Plymouth. This will help for visualisations where rather than plotting the lat/long per property we can group the building polygons for each cluster of properties.
#
# In this notebook we also calculate the distance per cluster to anchor properties.
#
# We:
# - Read data for Plymouth including which cluster each UPRN belongs to, the building polygons and the anchor properties.
# - Find which building polygon each UPRN sits within.
# - Join all the building polygons for UPRNs within the same cluster.
# - Output various stats about these joins - such as how many UPRNs are in a cluster, are in a building polygon, or aren't in a building polygon but others from the same cluster are.
# - Calculate the distance to anchor property per cluster.
# - At the end of the notebook there is some analysis to look at the clusters which overlap with the same building polygon as others as others.
#
#
# The most important file this notebook saves out is `s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/merged_uprns/per_cluster_merged_polygons.geojson` which contains the polygon per cluster and the distance from the nearest anchor property.

# %%
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import matplotlib.pyplot as plt

import os
import random

from asf_heat_pump_suitability.pipeline.prepare_features import anchor_properties

# %% [markdown]
# ## Read data for Plymouth
# - UPRNs and which cluster they are part of
# - Building polygons
# - Anchor properties

# %%
UPRNs_clustered = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/plymouth_residential_uprns_with_clusters/plymouth_residential_uprns_with_clusters.shp"
).to_crs(epsg=4326)

# %%
# Instead of all none clusters being assigned the -1 cluster, change this to a unique number - use the negative of the UPRN since this will be so much bigger, so won't overlap with genuine clusters
UPRNs_clustered["cluster"] = UPRNs_clustered.apply(
    lambda x: x["cluster"] if x["cluster"] >= 0 else -x["UPRN"], axis=1
)

# %%
assert (
    (UPRNs_clustered["cluster"] < 0).sum()
    + UPRNs_clustered[UPRNs_clustered["cluster"] >= 0]["cluster"].nunique()
) == UPRNs_clustered["cluster"].nunique()

# %%
print("\nLoading OS OpenMap Local - All buildings in SX region...")
os_openmap_building_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/OS OpenMap Local (ESRI Shape File) SX/data/SX_Building.shp"
).to_crs(epsg=4326)

# %%
print("\nFilter all buildings in SX region to just be in Plymouth...")
# There is probably a better way to do this using a Plymouth polygon - but this will do for now
building_footprints = os_openmap_building_gdf.cx[
    UPRNs_clustered["LONGITUDE"].min() : UPRNs_clustered["LONGITUDE"].max(),
    UPRNs_clustered["LATITUDE"].min() : UPRNs_clustered["LATITUDE"].max(),
]

# %%
print(
    f"There are {len(building_footprints)} buildings in Plymouth and {len(UPRNs_clustered)} UPRNs"
)

# %%
# Load anchor properties (for distance to anchor property per cluster)
anchor_properties_gdf = anchor_properties.load_gdf_and_process_poi()

# %% [markdown]
# ## Process and join data
#
# Find which building polygons each UPRN point sits within (if any)

# %%
UPRNs_clustered = gpd.GeoDataFrame(
    UPRNs_clustered,
    geometry=gpd.points_from_xy(UPRNs_clustered.LONGITUDE, UPRNs_clustered.LATITUDE),
    crs="EPSG:4326",
)

# %%
building_footprints["building_geom"] = (
    building_footprints.geometry
)  # Otherwise there are 2 geometry columns and this one gets dropped in the join

# %%
UPRNs_joined_buildings = gpd.sjoin(
    UPRNs_clustered,
    building_footprints,
    how="inner",
    predicate="intersects",
)

# %% [markdown]
# Finding out a few stats about this join

# %%
print(len(building_footprints))
print(len(UPRNs_clustered))
print(len(UPRNs_joined_buildings))

# %%
print(
    f"{UPRNs_joined_buildings['UPRN'].nunique()} out of {UPRNs_clustered['UPRN'].nunique()} unique UPRNs were found within building polygons. "
)

# %%
a = set(UPRNs_joined_buildings["cluster"].unique())
b = set(UPRNs_clustered["cluster"].unique())
print(
    f"{len(b.difference(a))} clusters out of {UPRNs_clustered['cluster'].nunique()} did not have at least one building polygon associated with it"
)


# %% [markdown]
# ## Merge building footprints
# For every cluster, find the union of the building polygons the UPRNs sit in


# %%
def merge_polygons(polygon_list):
    merged_polygon = gpd.GeoSeries(unary_union(polygon_list))
    return merged_polygon


# %%
# Example
# There will still be gaps between the geometries

cluster_number = 2905
UPRNs_joined_buildings_filtered = UPRNs_joined_buildings[
    UPRNs_joined_buildings["cluster"] == cluster_number
]

fig, ax = plt.subplots(1, 1)
gpd.GeoSeries(unary_union(UPRNs_joined_buildings_filtered["building_geom"])).plot(ax=ax)
UPRNs_joined_buildings_filtered[["geometry"]].plot(ax=ax, color="red")
plt.show()

# %% [markdown]
# - Merge polygons for clusters
# - Join the merged polygons to the UPRN data

# %%
merged_polygons_per_cluster = (
    UPRNs_joined_buildings.groupby("cluster")["building_geom"]
    .apply(lambda x: merge_polygons(x))
    .reset_index()
)

# %%
UPRNs_clustered_final = UPRNs_clustered.merge(
    merged_polygons_per_cluster, on="cluster", how="left"
)

# %% [markdown]
# #### Some stats

# %%
list_uprns_in_buildings = set(UPRNs_joined_buildings["UPRN"].tolist())

# %%
# Set a label to show that a UPRN has a building polygon
UPRNs_clustered_final["UPRN_joined_to_a_building_polygon"] = UPRNs_clustered_final[
    "UPRN"
].apply(lambda x: x in list_uprns_in_buildings)

# %%
UPRNs_clustered_final["UPRN_joined_to_a_building_polygon"].value_counts()

# %%
# Set a label to see if the cluster this UPRN is in has a polygon
UPRNs_clustered_final["cluster_polygon_found"] = UPRNs_clustered_final[
    "building_geom"
].notnull()

# %%
UPRNs_clustered_final["cluster_polygon_found"].value_counts()


# %%
def get_type(cluster, UPRN_joined_to_a_building_polygon, cluster_polygon_found):
    if cluster < 0:
        return "Not clustered"
    elif cluster_polygon_found:
        if UPRN_joined_to_a_building_polygon:
            return "UPRN and cluster in polygon"
        else:
            return "UPRN not in a polygon, but others in cluster are"
    else:
        if UPRN_joined_to_a_building_polygon:
            return "Unsual"  # Shouldnt happen
        else:
            return "UPRN nor any others in cluster are in any polygons"


# %%
# Set a single label to combine the different cases for this UPRN being in a polygon
UPRNs_clustered_final["Type"] = UPRNs_clustered_final.apply(
    lambda x: get_type(
        x["cluster"], x["UPRN_joined_to_a_building_polygon"], x["cluster_polygon_found"]
    ),
    axis=1,
)

# %%
UPRNs_clustered_final["Type"].value_counts()

# %%
UPRNs_clustered_final["Type"].value_counts(normalize=True)

# %% [markdown]
# ### Save data
# #### Per UPRN which cluster and the extra info about whether it was in a polygon
#

# %%
UPRNs_clustered_final.drop(columns=["building_geom", "geometry"]).to_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/merged_uprns/per_uprn_merged_polygons.csv",
)

# %% [markdown]
# #### Per building footprint
#

# %% [markdown]
# ##### 1. Add the distance to an anchor property for each cluster polygon

# %%
merged_polygons_per_cluster_27700 = merged_polygons_per_cluster.copy(deep=True)
merged_polygons_per_cluster_27700.crs = "EPSG:4326"
merged_polygons_per_cluster_27700 = merged_polygons_per_cluster_27700.to_crs(epsg=27700)

# %%
# If not the same crs, convert both to 27700
assert anchor_properties_gdf.crs == merged_polygons_per_cluster_27700.crs

# %%
# Add distance from cluster polygon to anchor load

# Replace polygon geometry with centroid point (if you want to keep the polygons, i suggest copying the gdf first)
merged_polygons_per_cluster_27700["geometry"] = merged_polygons_per_cluster_27700[
    "building_geom"
].centroid

# Get distance from cluster centroid to nearest anchor property
merged_polygons_per_cluster_27700 = merged_polygons_per_cluster_27700.sjoin_nearest(
    anchor_properties_gdf[["geometry"]],
    how="left",
    distance_col="distance_from_anchor_property_m",
).drop(columns="index_right")

# %%
print(len(merged_polygons_per_cluster))
print(len(merged_polygons_per_cluster_27700))
print(merged_polygons_per_cluster_27700["cluster"].nunique())

# %%
# I'm not sure why but a very small number of clusters keeps getting duplicated in this sjoin, the results seem to be the same though, so just deduplicate

merged_polygons_per_cluster_27700.drop_duplicates(subset="cluster", inplace=True)
print(len(merged_polygons_per_cluster_27700))

# %%
## Merge this newly calculated distance from anchor property with the lat/long version of the cluster polygons
merged_polygons_per_cluster = merged_polygons_per_cluster.merge(
    merged_polygons_per_cluster_27700[["cluster", "distance_from_anchor_property_m"]],
    on="cluster",
)

# %% [markdown]
# ##### 2. Save

# %%
## The cluster polygons
gpd.GeoDataFrame(merged_polygons_per_cluster, geometry="building_geom").to_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/merged_uprns/per_cluster_merged_polygons.geojson",
    driver="GeoJSON",
)

# %% [markdown]
# ##### 3. Save all polygons for Plymouth, and information about whether UPRNs were in it (useful for debugging)

# %%
# Get the building polygons with no UPRNs in
building_footprints_no_uprn = building_footprints[
    ~building_footprints["building_geom"].isin(UPRNs_joined_buildings["building_geom"])
]

# %%
# The polygons per cluster + the polygons with no UPRNs
all_building_polygons = pd.concat(
    [merged_polygons_per_cluster, building_footprints_no_uprn[["building_geom"]]]
)

# %%
# A label about which type of polygon it is (in terms of UPRN clusters within it)
all_building_polygons["Type"] = all_building_polygons["cluster"].apply(
    lambda x: (
        "Has UPRNS but no cluster"
        if x < 0
        else ("No UPRN" if pd.isnull(x) else "Has UPRN from a cluster")
    )
)

# %%
gpd.GeoDataFrame(all_building_polygons, geometry="building_geom").to_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/merged_uprns/per_cluster_merged_polygons_inc_no_uprns.geojson",
    driver="GeoJSON",
)

# %% [markdown]
# #### Some stats

# %%
print(len(merged_polygons_per_cluster))
print(len(building_footprints_no_uprn))

# %%
all_building_polygons["Type"].value_counts()

# %% [markdown]
# ## Testing: Some clusters have overlapped with the same building polygon as others
#
# This section is just checking things about the data, no saving is done

# %% [markdown]
# How often does this happen?

# %%
buildings_with_clusters = all_building_polygons[
    all_building_polygons["Type"] == "Has UPRN from a cluster"
]

# %%
print(
    f"There are {buildings_with_clusters['building_geom'].nunique()} unique cluster polygons for {len(buildings_with_clusters)} clusters"
)

# %%
duplicated_cluster_polygons = buildings_with_clusters[
    buildings_with_clusters.duplicated(subset="building_geom", keep=False)
]

# %%
print(
    f"There are {len(duplicated_cluster_polygons)} clusters in the same {duplicated_cluster_polygons['building_geom'].nunique()} polygons"
)

# %%
cluster_number = 200
joined_filtered = UPRNs_joined_buildings[
    UPRNs_joined_buildings["cluster"] == cluster_number
]

fig, ax = plt.subplots(1, 1)
gpd.GeoSeries(unary_union(joined_filtered["building_geom"])).plot(ax=ax)
joined_filtered[["geometry"]].plot(ax=ax, color="red")
plt.show()

# %%
cluster_number = 205
joined_filtered = UPRNs_joined_buildings[
    UPRNs_joined_buildings["cluster"] == cluster_number
]

fig, ax = plt.subplots(1, 1)
gpd.GeoSeries(unary_union(joined_filtered["building_geom"])).plot(ax=ax)
joined_filtered[["geometry"]].plot(ax=ax, color="red")
plt.show()

# %%
clusters_in_dupes = (
    duplicated_cluster_polygons.groupby("building_geom")["cluster"]
    .unique()
    .reset_index()
)

cmap = plt.cm.get_cmap("hsv", 500)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

gpd.GeoSeries(clusters_in_dupes["building_geom"]).plot(ax=ax)

for cluster_list in clusters_in_dupes["cluster"]:
    for r, cluster_number in enumerate(cluster_list):
        joined_filtered = UPRNs_joined_buildings[
            UPRNs_joined_buildings["cluster"] == cluster_number
        ]
        joined_filtered[["geometry"]].plot(
            ax=ax, color=cmap(random.randint(0, 500)), alpha=0.3
        )

plt.show()

# %%
