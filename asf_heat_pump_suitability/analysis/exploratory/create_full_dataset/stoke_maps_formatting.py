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
# ## Format and merge Stoke data into a format for plotting our maps
#
# In this notebook we:
# - Filter the Plymouth cluster, feasibility and suitability data to the Stoke ward
# - Bring in some extra datasets; HN pilot zones, green space and anchor properties
# - Format the data for plotting in Flourish.

# %%
import geopandas as gpd
import pandas as pd
import random

from asf_heat_pump_suitability.analysis.exploratory.create_full_dataset import (
    stoke_getters,
)
import config

# %% [markdown]
# ## Import data

# %%
cluster_polygons = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/merged_uprns/per_cluster_merged_polygons.geojson"
).to_crs(epsg=4326)

# %%
feasibility_scoring_data = pd.read_parquet(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/plymouth_feasibility_scoring.parquet"
)

suitability_categorisation_data = pd.read_parquet(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/plymouth_suitability_categorisation.parquet"
)

# %%
# Lat/long of each UPRN
UPRNs_clustered = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/plymouth_residential_uprns_with_clusters/plymouth_residential_uprns_with_clusters.shp"
).to_crs(epsg=4326)

# %%
# Load per UPRN features
plymouth_uprn_data = pd.read_parquet(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/results/plymouth_features_full_with_clusters.parquet",
    columns=[
        "UPRN",
        "in_listed_building",
        "in_cons_area",
        "in_hn",
        "filled_off_gas",
        "imd_decile",
        "use_garden_area_m2",
        "use_community_heating",
        "use_property_type",
        "use_tenure",
        "cluster",
    ],
)

# %%
stoke_ward = stoke_getters.load_stoke_bound().to_crs(epsg=4326)

# %%
stoke_greenspace_df_formatted = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_greenspace.geojson",
).to_crs(epsg=4326)

hnz_plymouth = gpd.read_file(
    config["data"]["geodata"]["heat_network_zones"]["plymouth"],
    columns=["geometry", "Type"],
).to_crs(epsg=4326)

# %% [markdown]
# ## Format per property data
#
# - Merge feature data with lat/long coordinates
# - Filter for Stoke
# - Rename columns
# - At jitter to lat/long coordinates to allow for points to be plotted over one another
# - Save

# %%
plymouth_uprn_data = plymouth_uprn_data.merge(
    UPRNs_clustered[["UPRN", "LATITUDE", "LONGITUDE", "geometry"]],
    on="UPRN",
    how="left",
)

# %%
stoke_uprn_data = gpd.GeoDataFrame(plymouth_uprn_data, geometry="geometry").sjoin(
    stoke_ward[["geometry"]],
    how="inner",
    predicate="intersects",
)

# %%
stoke_uprn_data.rename(
    columns={
        "in_listed_building": "In listed building",
        "in_cons_area": "In conservation zone",
        "in_hn": "In a HN pilot zone",
        "filled_off_gas": "Off gas",
        "imd_decile": "IMD decile",
        "use_garden_area_m2": "Garden area (m2)",
        "use_community_heating": "On communal heating",
        "use_property_type": "Property Type",
        "use_tenure": "Tenure",
    },
    inplace=True,
)

# %%
stoke_uprn_data["LATITUDE_jit"] = stoke_uprn_data["LATITUDE"] + [
    random.uniform(-0.00001, 0.00001) for i in range(len(stoke_uprn_data))
]
stoke_uprn_data["LONGITUDE_jit"] = stoke_uprn_data["LONGITUDE"] + [
    random.uniform(-0.0001, 0.0001) for i in range(len(stoke_uprn_data))
]

# %%
stoke_uprn_data["Garden area (m2)"] = stoke_uprn_data["Garden area (m2)"].round(2)
stoke_uprn_data["IMD decile"] = stoke_uprn_data["IMD decile"].round(0)

# %% [markdown]
# ## Checks
# There are 4 clusters in the suitability data but not feasibility or cluster data? No idea why?
#
# Ignore these 4.

# %%
print(len(cluster_polygons))
print(len(feasibility_scoring_data))
print(len(suitability_categorisation_data))

# %%
a = set(suitability_categorisation_data["cluster"].tolist())
b = set(feasibility_scoring_data["cluster"].tolist())

# %%
a.difference(b)

# %%
suitability_categorisation_data[
    suitability_categorisation_data["cluster"].isin(list(a.difference(b)))
]

# %%
suitability_categorisation_data["most_suitable_tech"].value_counts()

# %% [markdown]
# ## Convert I ASHP to C ASHP since a collective scheme could span across the entirity of Plymouth
#
# Should eventually change this in `assign_cluster_suitability_and_feasibility.py`

# %%
suitability_categorisation_data.loc[
    suitability_categorisation_data["most_suitable_tech"] == "individual_ashp",
    "most_suitable_tech",
] = "collective_ashp"

suitability_categorisation_data["most_suitable_tech"].value_counts()

# %% [markdown]
# ## Rename columns and some values in the cluster, feasibility and suitability datasets

# %%
cluster_polygons.rename(
    columns={"distance_from_anchor_property_m": "Distance from anchor property (m)"},
    inplace=True,
)

# %%
feasibility_scoring_data.rename(
    columns={
        "perc_owner_occupied": "% owner occupied",
        "perc_imd_decile_above_avg": "% IMD decile above average",
        "perc_on_gas": "% on gas",
        "perc_in_listed_building": "% listed buildings",
        "perc_not_in_listed_building": "% not listed buildings",
        "perc_in_conservation_area": "% in conservation area",
        "perc_not_in_conservation_area": "% not in conservation area",
        "perc_social_housing": "% social housing",
        "perc_flats": "% flats",
        "perc_on_communal_heating": "% on communal heating",
        "perc_has_outdoor_space": "% with > 50m2 outdoor space",
        "perc_in_heat_network_zone": "% in a HN pilot zone",
        "perc_close_to_city_centre": "% close to a city centre",
        "perc_close_to_anchor_loads": "% < 500 m from an anchor load",
        "cluster_size": "Number of residential homes in cluster",
        "collective_ashp_feasibility": "ASHP feasibility",
        "hn_feasibility": "Networked HPs feasibility",
        "sgl_feasibility": "Communal heat source feasibility",
    },
    inplace=True,
)

# %%
suitability_categorisation_data.rename(
    columns={
        "in_heat_network_zone": "In a HN pilot zone",
        "in_city_centre": "In city centre",
        "total_outdoor_space": "Total outdoor space (m2)",
        "has_outdoor_space": "Has > 50m2 of outdoor space?",
        "most_suitable_tech": "Most suitable technology",
    },
    inplace=True,
)

# %%
suitability_categorisation_data[
    "Most suitable technology"
] = suitability_categorisation_data["Most suitable technology"].map(
    {
        "collective_ashp": "ASHP",
        "shared_ground_loop": "Communal heat source",
        "heat_network": "Networked heat pumps",
    }
)

# %% [markdown]
# ## Combine datasets and save
# - Suitability for all Plymouth + Stoke boundary
# - Feasibility for just Stoke

# %%
merged_data = cluster_polygons.merge(feasibility_scoring_data, on="cluster", how="left")
merged_data = merged_data.merge(
    suitability_categorisation_data, on="cluster", how="left"
)

# %% [markdown]
# ### Stoke suitability map
#
# This map will include:
#
# Regions (`s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_plot_data_all_region_types.geojson`):
# - Clusters, suitability and feasibility features and scores
# - Green space
# - Boundary for stoke
# - HN zones
# - Just most suitable tech data
#
# Points (`s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_anchorloads.geojson`)
# - Anchor properties (not created here)

# %%
stoke_cluster_polygons = cluster_polygons.sjoin(
    stoke_ward[["geometry"]],
    how="inner",
    predicate="intersects",
)

# %%
len(stoke_cluster_polygons)

# %%
stoke_suit_plot_data = stoke_cluster_polygons.merge(
    suitability_categorisation_data, on="cluster", how="left"
)

# %%
stoke_suit_plot_data["Type"] = stoke_suit_plot_data["Most suitable technology"]

# %%
stoke_ward["Type"] = "Stoke boundary"
hnz_plymouth["Type"] = "DESNZ HN zone"

# %%
# Helpful for Flourish tooltips
hnz_plymouth["hn_zone"] = 1
stoke_greenspace_df_formatted["greenspace"] = 1
stoke_greenspace_df_formatted["Type"] = "Greenspace"

# %%
stoke_plot_data_all_region_types = pd.concat(
    [
        hnz_plymouth,
        stoke_greenspace_df_formatted,
        stoke_suit_plot_data,
    ]
)

# %% [markdown]
# ### Rename columns and values

# %%
gpd.GeoDataFrame(stoke_plot_data_all_region_types, geometry="geometry").to_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_plot_data_all_region_types.geojson",
    driver="GeoJSON",
)

# %% [markdown]
# ## Stoke feasibility map
#
# Regions (`s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_merged_data.geojson`)
# - Feasibility scores and features per cluster
#
# Points (`s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_uprn_data.csv`)
# - UPRNs with features and a bit of jitter added to lat/long

# %%
stoke_merged_data = merged_data.sjoin(
    stoke_ward[["geometry"]],
    how="inner",
    predicate="intersects",
)

# %%
gpd.GeoDataFrame(stoke_merged_data.round(1), geometry="geometry").to_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_merged_data.geojson",
    driver="GeoJSON",
)

# %%
stoke_uprn_data.to_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/stoke_uprn_data.csv",
)

# %%
