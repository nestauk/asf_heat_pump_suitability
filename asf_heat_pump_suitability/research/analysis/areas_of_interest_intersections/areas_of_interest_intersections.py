# %% [markdown]
# ## Areas of interest intersections
#
# This notebook aggregates cluster data and UPRN data from the Local Heat Planning Tool that intersect with four given areas of interest.

# %%
import os

from datetime import datetime

import geopandas as gpd
import polars as pl

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from asf_heat_pump_suitability.pipeline.transform import uprns
from asf_heat_pump_suitability import PROJECT_DIR

# %%
date = datetime.today().strftime("%Y%m%d")

# %% [markdown]
# # Get intersecting cluster data

# %%
# Load areas of interest
aoi_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/plymouth_inputs/Opportunity_areas_PCC_x_PCH.kml"
)

# Load clusters
clusters_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth/plymouth_clusters_contextual_features_5m.geojson"
)

# Convert clusters to same CRS
clusters_gdf["geometry"] = clusters_gdf.force_2d()
clusters_gdf = clusters_gdf.to_crs(aoi_gdf.crs)

assert aoi_gdf.crs == clusters_gdf.crs

# Get clusters intersecting with areas of interest
keep_cols = ["Name"] + list(clusters_gdf.columns)
coi_gdf = clusters_gdf.sjoin(aoi_gdf, how="inner", predicate="intersects")[keep_cols]
coi_gdf = coi_gdf[coi_gdf["assigned_tech"] != "DESNZ_HNZ"]

# Get counts of assigned tech per area of interest
assigned_techs_per_area_gdf = (
    coi_gdf.groupby(["Name", "assigned_tech"])
    .agg(n_UPRNs=("n_UPRNs", "sum"))
    .reset_index()
    .pivot(columns="assigned_tech", index="Name", values="n_UPRNs")
)
assigned_techs_per_area_gdf.columns.name = None
assigned_techs_per_area_gdf = (
    assigned_techs_per_area_gdf.rename(
        columns={
            col: f"n_uprns_assigned_{col.replace(" ", "_")}"
            for col in coi_gdf["assigned_tech"]
        }
    )
    .reset_index()
    .fillna(0)
)

# Columns to be summed
sum_cols = [
    "attachment_detached",
    "attachment_end_terrace",
    "attachment_flat",
    "attachment_mid_terrace",
    "attachment_semi_detached",
    "attachment_null",
    "tenure_null",
    "tenure_owner_occupied",
    "tenure_rental_(private)",
    "tenure_rental_(social)",
    "current_energy_rating_a",
    "current_energy_rating_b",
    "current_energy_rating_c",
    "current_energy_rating_d",
    "current_energy_rating_e",
    "current_energy_rating_f",
    "current_energy_rating_g",
    "current_energy_rating_null",
    "n_UPRNs",
    "n_uprns_in_listed_building",
    "n_uprns_solar_pv",
    "n_uprns_off_gas",
    "n_uprns_in_protected_area",
    "n_uprns_within_1500m_of_coastline",
    "n_uprns_in_hn_zone",
    "n_uprns_in_city_centre",
]

# Columns to be averaged
avg_cols = [
    "median_estimated_energy_consumption_12_months_kwh_per_m2",
    "median_outdoor_space_m2",
]

# Aggregate columns by summing / averaging and merge into one df
sum_coi_gdf = coi_gdf[["Name"] + sum_cols].groupby("Name").agg("sum").reset_index()
avg_coi_gdf = (
    coi_gdf[["Name"] + avg_cols]
    .groupby("Name")
    .agg("mean")
    .rename(columns={col: f"avg_{col}" for col in avg_cols})
    .reset_index()
)
final_df = assigned_techs_per_area_gdf.merge(sum_coi_gdf, how="left", on="Name").merge(
    avg_coi_gdf, how="left", on="Name"
)

# Save dataframe to file
filename = f"{date}_plymouth_cluster_data_intersecting_opportunity_areas_pcc_x_pch.csv"
fpath = os.path.join(PROJECT_DIR, "outputs", "data", filename)
final_df.to_csv(fpath)

# %%
# Plot the clusters that intersect with the areas of interest

fig, axes = plt.subplots(2, 2, figsize=(12, 5 * 2))
axes = axes.flatten()

for i, n in enumerate(coi_gdf["Name"].unique()):
    ax = axes[i]
    coi_gdf[coi_gdf["Name"] == n].dissolve(by="Name").plot(
        ax=ax, alpha=0.5, label="Overlapping clusters"
    )
    aoi_gdf[aoi_gdf["Name"] == n].plot(
        ax=ax, color="red", alpha=0.5, label="Areas of interest"
    )
    ax.set_title(n)
    if i == 1:
        ax.legend(
            handles=[
                mpatches.Patch(color="red", alpha=0.5, label="Areas of interest"),
                mpatches.Patch(color="blue", alpha=0.5, label="Overlapping clusters"),
            ],
            loc="upper right",
            bbox_to_anchor=(1.5, 1),
        )

filename = (
    f"{date}_plymouth_cluster_data_intersecting_opportunity_areas_pcc_x_pch_maps.png"
)
fpath = os.path.join(PROJECT_DIR, "outputs", "figures", filename)
fig.savefig(fpath)

# %% [markdown]
# ## Get intersecting UPRN data

# %%
# Load areas of interest
aoi_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/plymouth_inputs/Opportunity_areas_PCC_x_PCH.kml"
)

# Load UPRNs in Plymouth with features
plymouth_features_df = pl.read_parquet(
    "s3://asf-local-heat-planning-tool/outputs/data/plymouth/plymouth_with_features.parquet"
)

# Convert to geodataframe with point geometries per UPRN
plymouth_features_gdf = uprns.generate_gdf_uprn_coords(plymouth_features_df)
analysis_gdf = plymouth_features_gdf.to_crs(epsg=4326)
assert aoi_gdf.crs == analysis_gdf.crs

# Get UPRNs intersecting with area of interest
uprns_of_interest_df = pl.from_pandas(
    analysis_gdf.sjoin(aoi_gdf, how="inner", predicate="intersects").drop(
        columns="geometry"
    )
)

# Create dummy columns for summing
dummy_cols = ["ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]
uprns_of_interest_df = uprns_of_interest_df.to_dummies(columns=dummy_cols).to_pandas()

avg_cols = [
    "max_contiguous_outdoor_space_area_m2",
    "ENERGY_CONSUMPTION_CURRENT",
]

sum_cols = [
    "in_hn_zone",
    "in_city_centre",
    "has_solar_pv",
    "in_listed_building",
    "off_gas",
    "within_1500m_coastline",
    "in_protected_area",
    "ATTACHMENT_Detached",
    "ATTACHMENT_End-Terrace",
    "ATTACHMENT_Flat",
    "ATTACHMENT_Mid-Terrace",
    "ATTACHMENT_Semi-Detached",
    "ATTACHMENT_null",
    "TENURE_null",
    "TENURE_owner-occupied",
    "TENURE_rental (private)",
    "TENURE_rental (social)",
    "CURRENT_ENERGY_RATING_A",
    "CURRENT_ENERGY_RATING_B",
    "CURRENT_ENERGY_RATING_C",
    "CURRENT_ENERGY_RATING_D",
    "CURRENT_ENERGY_RATING_E",
    "CURRENT_ENERGY_RATING_F",
    "CURRENT_ENERGY_RATING_G",
    "CURRENT_ENERGY_RATING_null",
]

# Aggregate columns by summing / averaging and merge into one df
sum_df = (
    uprns_of_interest_df[["Name"] + sum_cols]
    .groupby("Name")
    .agg("sum")
    .rename(columns={col: f"n_UPRNs_{col}" for col in sum_cols})
    .reset_index()
)
avg_df = (
    uprns_of_interest_df[["Name"] + avg_cols]
    .groupby("Name")
    .agg("median")
    .rename(columns={col: f"median_{col}" for col in avg_cols})
    .reset_index()
)
count_df = (
    uprns_of_interest_df[["Name", "UPRN"]]
    .groupby("Name")
    .agg("count")
    .rename(columns={"UPRN": "n_UPRNs"})
)
final_df = count_df.merge(sum_df, how="left", on="Name").merge(
    avg_df, how="left", on="Name"
)

# Save final dataframe
filename = f"{date}_plymouth_cluster_data_intersecting_UPRNs_pcc_x_pch.csv"
fpath = os.path.join(PROJECT_DIR, "outputs", "data", filename)
final_df.to_csv(fpath)
