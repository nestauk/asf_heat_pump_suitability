# %% [markdown]
# ### **Investigating areas in Plymouth LA with existing/planned heat network zones or areas which are city areas**

# %% [markdown]
# This is an exploratory notebook where:
# 1. Existing/planned heat network zones in Plymouth are inspected visually and the % of UPRNs in a zone is calculated.
# 2. Which spatial signature types from the Spatial Signatures Framework are in existing/planned heat network zones. Within each existing/planned heat network zone in Plymouth, each type is identified.
# 3. A subset of spatial signature types selected to represent city centre areas are examined and their spatial and UPRN coverage is compared with that of existing/planned heat network zones.

# %%
import pandas as pd
import geopandas as gpd
import polars as pl
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import (
    base_getters,
    load_geodata,
    load_boundaries,
)
from asf_heat_pump_suitability.pipeline.transform import (
    uprns,
    heat_network_zones,
    city_centres,
)

# %% [markdown]
# Loading data

# %%
# OSOpen UPRNS filtered for Plymouth only
plymouth_uprns_df = base_getters.load_df_from_s3(
    "s3://asf-heat-pump-suitability/dump/plymouth_OSOpen_UPRNs_filtered.parquet"
)
plymouth_uprns_gdf = uprns.generate_gdf_uprn_coords(plymouth_uprns_df)

# %%
# Plymouth residential UPRNs
plymouth_residential_uprns_df = base_getters.load_df_from_s3(
    config["data"]["processed"]["plymouth_residential_uprns"]
)
plymouth_residential_uprns_gdf = uprns.generate_gdf_uprn_coords(
    plymouth_residential_uprns_df
)

# %%
# Plymouth LA boundary
plymouth_la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
    select_las="plymouth"
)

# %%
# Existing/planned heat network zones in plymouth
plymouth_hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(
    local_authority="plymouth"
)

# %%
# Simplified form of the spatial signatures gb dataset
spatial_signatures_gb_simplified_gdf = load_geodata.load_gdf_spatial_signatures_gb(
    detail_level="simplified"
)

# %%
# Confirming spatial signature types
print("Spatial signatures:")
counter = 1
for type in spatial_signatures_gb_simplified_gdf["type"].unique():
    print(f"{counter}. {type}")
    counter += 1

# %%
# CRS checks
gdfs = [
    plymouth_uprns_gdf,
    plymouth_residential_uprns_gdf,
    plymouth_la_boundaries_gdf,
    plymouth_hn_zones_gdf,
    spatial_signatures_gb_simplified_gdf,
]
len({gdf.crs for gdf in gdfs}) == 1

# %% [markdown]
# #### **Inspecting existing/planned heat network zones**

# %%
# Inspecting each HNZ
plymouth_boundary = plymouth_la_boundaries_gdf.plot(
    color="white", edgecolor="black"
)  # Plymouth LA boundary

plymouth_hn_zones_gdf.plot(
    ax=plymouth_boundary,
    edgecolor="blue",
    legend=True,
    column="ZoneID",
)  # Existing heat network zones

# %%
# Label residential UPRNs in HNZ
plymouth_uprns_hnz_labelled_df = heat_network_zones.label_gdf_heat_network_zone_uprns(
    uprn_gdf=plymouth_residential_uprns_gdf,
    hn_zone_gdf=plymouth_hn_zones_gdf,
    usecols=["ZoneID"],
)

# Filter only in HNZ
plymouth_hn_zone_uprns_df = plymouth_uprns_hnz_labelled_df.filter(
    pl.col("in_hn_zone") == True
)
plymouth_hn_zone_uprns_gdf = uprns.generate_gdf_uprn_coords(plymouth_hn_zone_uprns_df)

# %%
# Inspecting layers of UPRNs, boundaries and HNZ

# Layers
plymouth_boundary = plymouth_la_boundaries_gdf.plot(color="white", edgecolor="black")

plymouth_uprns_gdf.plot(
    ax=plymouth_boundary, color="grey", markersize=2, label="All Plymouth UPRNs"
)

plymouth_residential_uprns_gdf.plot(
    ax=plymouth_boundary, color="red", markersize=2, label="Residential UPRNs only"
)

plymouth_hn_zones_gdf.plot(
    ax=plymouth_boundary, facecolor="none", edgecolor="blue", linewidth=2
)

plymouth_hn_zone_uprns_gdf.plot(
    ax=plymouth_boundary, color="yellow", markersize=2, label="Residential UPRNs in HNZ"
)

# Manual legend patches for boundaries
boundary_patch = mpatches.Patch(
    facecolor="white", edgecolor="black", label="Plymouth LA boundary"
)

hn_zone_patch = mpatches.Patch(
    facecolor="none", edgecolor="blue", label="Existing heat network zones"
)

# Combine auto and manual labels
handles, labels = plymouth_boundary.get_legend_handles_labels()

new_handles = handles + [boundary_patch, hn_zone_patch]
new_labels = labels + ["Plymouth LA boundary", "Existing heat network zones"]

plt.legend(
    handles=new_handles,
    labels=new_labels,
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    title="Layers",
)

plt.show()

# %%
# What % of residential UPRNs are in existing HNZ?
prop_in_hn_zone = plymouth_uprns_hnz_labelled_df["in_hn_zone"].mean()

print(f"Proportion in HN zone: {prop_in_hn_zone:.3f}")

# %% [markdown]
# #### **Investigating which spatial signature types are in existing heat network zones**

# %%
# Spatial signatures in Plymouth LA boundaries only
spatial_signatures_plymouth_gdf = spatial_signatures_gb_simplified_gdf.sjoin(
    plymouth_la_boundaries_gdf, how="inner", predicate="intersects"
).drop(columns="index_right")

# %%
# colour for spatial signature types
unique_types = spatial_signatures_plymouth_gdf["type"].unique()
colors = plt.cm.tab20.colors
color_dict = {t: colors[i % len(colors)] for i, t in enumerate(unique_types)}


fig, ax = plt.subplots(figsize=(15, 7.5))

# plot spatial signatures polygons with mapped colors
spatial_signatures_plymouth_gdf.plot(
    ax=ax, color=spatial_signatures_plymouth_gdf["type"].map(color_dict)
)

# overlay Plymouth LA boundaries
plymouth_la_boundaries_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

# overlay Plymouth heat network zones
plymouth_hn_zones_gdf.plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=1)

# manual legend for spatial signature types
legend_handles = [mpatches.Patch(color=color_dict[t], label=t) for t in unique_types]

# lines for boundaries and heat network zones
legend_handles += [
    Line2D([0], [0], color="black", lw=2, label="Plymouth LA boundary"),
    Line2D([0], [0], color="blue", lw=2, label="Existing heat network zones"),
]

ax.legend(
    handles=legend_handles,
    title="Spatial Signature Type",
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
)

plt.show()

# %% [markdown]
# 05/12/25: There is no clear consistency in the spatial signature types in existing heat network zones.

# %% [markdown]
# Inspecting existing HNZ one at a time:

# %%
plymouth_hn_zones_with_labels_gdf = gpd.sjoin(
    plymouth_hn_zones_gdf,
    spatial_signatures_plymouth_gdf[["type", "geometry"]],
    how="left",
    predicate="within",
)

plymouth_hn_zones_with_labels_agg_gdf = (
    plymouth_hn_zones_with_labels_gdf.groupby("ZoneID")["type"]
    .apply(lambda x: list(x.dropna().unique()))
    .reset_index()
)


# %%
# helper function to visually inspect each HNZ
def inspect_spatial_signatures_in_hn_zone(zone_id: str):
    zone_gdf = plymouth_hn_zones_gdf[plymouth_hn_zones_gdf["ZoneID"] == zone_id]

    # Add HN zone notes
    just_text_raw = str(zone_gdf.Just.values[0])
    just_text = "\n".join(textwrap.wrap(just_text_raw, width=90))

    spatial_signatures_gdf = spatial_signatures_plymouth_gdf.sjoin(
        zone_gdf, how="inner", predicate="intersects"
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    base = spatial_signatures_gdf.plot(column="type", legend=True, ax=ax)
    zone_gdf.plot(ax=base, facecolor="none", edgecolor="red")
    ax.set_title(f"Zone: {zone_id}", fontsize=12)

    fig.text(
        0.5,
        0.01,
        just_text,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    return ax


# %%
for zone_id in plymouth_hn_zones_with_labels_agg_gdf["ZoneID"].unique():
    inspect_spatial_signatures_in_hn_zone(zone_id)

# %% [markdown]
# #### **Testing urban spatial signatures to indicate city centre areas**

# %% [markdown]
# Labelling original residential UPRNs

# %%
# Test set of signature types
city_centre_types = [
    "Hyper concentrated urbanity",
    "Concentrated urbanity",
    "Metropolitan urbanity",
    "Regional urbanity",
    "Local urbanity",
    "Dense urban neighbourhoods",
]

# %%
plymouth_uprns_spatial_signature_labelled_df = (
    city_centres.label_gdf_city_centre_spatial_signatures_uprns(
        uprn_gdf=plymouth_residential_uprns_gdf,
        spatial_signatures_gdf=spatial_signatures_gb_simplified_gdf,
        types=city_centre_types,
    )
)

# %%
# Process filtered UPRNs for plotting
plymouth_uprns_spatial_signature_df = (
    plymouth_uprns_spatial_signature_labelled_df.filter(
        pl.col("in_city_centre") == True
    )
)
plymouth_uprns_spatial_signature_gdf = uprns.generate_gdf_uprn_coords(
    plymouth_uprns_spatial_signature_df
)

# %%
# Filter spatial signatures in Plymouth to selected types
selected_spatial_signatures_plymouth_gdf = spatial_signatures_plymouth_gdf[
    spatial_signatures_plymouth_gdf["type"].isin(city_centre_types)
]

# %%
# Inspect selected spatial signatures in relation to existing/planned HNZ

# colour for spatial signature types
unique_types = selected_spatial_signatures_plymouth_gdf["type"].unique()
colors = plt.cm.tab20.colors
color_dict = {t: colors[i % len(colors)] for i, t in enumerate(unique_types)}


fig, ax = plt.subplots(figsize=(15, 7.5))

# plot spatial signatures polygons with mapped colors
selected_spatial_signatures_plymouth_gdf.plot(
    ax=ax, color=selected_spatial_signatures_plymouth_gdf["type"].map(color_dict)
)

# overlay Plymouth LA boundaries
plymouth_la_boundaries_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

# overlay Plymouth heat network zones
plymouth_hn_zones_gdf.plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=1)

# manual legend for spatial signature types
legend_handles = [mpatches.Patch(color=color_dict[t], label=t) for t in unique_types]

# lines for boundaries and heat network zones
legend_handles += [
    Line2D([0], [0], color="black", lw=2, label="Plymouth LA boundary"),
    Line2D([0], [0], color="blue", lw=2, label="Existing heat network zones"),
]

ax.legend(
    handles=legend_handles,
    title="Spatial Signature Type",
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
)

plt.show()

# %% [markdown]
# 05/12/25: Only 3 of the 6 city centre spatial signature types are present within Plymouth LA boundaries (Dense urban neighbourhoods, Local urbanity and Regional urbanity). Only the existing HNZ PLYM-011 contain any of these city centre types. The remaining 7 existing HN zones are in a wide range of other spatial signature types.

# %% [markdown]
# **Comparison of zones by spatial coverage**

# %%
# Comparing total spatial overlap
overlap = gpd.overlay(
    plymouth_hn_zones_gdf, selected_spatial_signatures_plymouth_gdf, how="intersection"
)
overlap_area = overlap.geometry.area.sum()

# % overlap of each layer
area_hnz = plymouth_hn_zones_gdf.geometry.area.sum()
area_selected = selected_spatial_signatures_plymouth_gdf.geometry.area.sum()

percent_hnz_covered_by_selected = (overlap_area / area_hnz) * 100
percent_selected_covered_by_hnz = (overlap_area / area_selected) * 100

# summary table
summary_df = pd.DataFrame(
    {
        "Area (m²)": [area_hnz, area_selected, overlap_area],
        "% of HNZ covered by selected signatures": [
            f"{percent_hnz_covered_by_selected:.1f}%",
            "",
            "",
        ],
        "% of selected signatures covered by HNZ": [
            "",
            f"{percent_selected_covered_by_hnz:.1f}%",
            "",
        ],
    },
    index=["Existing HNZ", "Selected signatures", "Overlap"],
)

summary_df

# %% [markdown]
# 05/12/25: Computing the spatial overlap, the **city centre types cover 35% of the area covered by existing heat network zones**. Conversely, the **existing heat network zones cover 83% of the area covered by city centre types**.
# - 35% metric -> The selected signature types only partially align with the existing heat network coverage; most of the current HNZ lie outside "city centre" areas. This indicates that using "city centre" areas is not able to capture all areas that have been deemed suitable for a heat network in reality.
# - 83% metric -> Most of the areas identified as city centre are already part of the existing heat network.
# - Using city centre areas as proxies for heat network suitability: most areas selected are part of an existing HNZ (high precision) - but many existing HNZ areas outside these urban centres are missed, since HNZ are not chosen based on urban location alone.

# %%
# Inspecting layers of UPRNs, boundaries and spatial signatures

# Layers
plymouth_boundary = plymouth_la_boundaries_gdf.plot(color="white", edgecolor="grey")


plymouth_residential_uprns_gdf.plot(
    ax=plymouth_boundary, color="#8B0000", markersize=2, label="Residential UPRNs only"
)

plymouth_hn_zones_gdf.plot(
    ax=plymouth_boundary,
    facecolor="none",
    edgecolor="#1f77b4",
    linewidth=2,
    alpha=0.8,
    label="HN Zones",
)

plymouth_hn_zone_uprns_gdf.plot(
    ax=plymouth_boundary,
    color="#FFD700",
    markersize=3,
    label="Residential UPRNs in HNZ",
)

selected_spatial_signatures_plymouth_gdf.plot(
    ax=plymouth_boundary,
    facecolor="none",
    edgecolor="#800080",
    linewidth=2,
    label="Selected Spatial Signatures",
)

plymouth_uprns_spatial_signature_gdf.plot(
    ax=plymouth_boundary,
    color="#228B22",
    markersize=3,
    label="Residential UPRNs in selected spatial signatures",
)

# Manual legend patches for boundaries
boundary_patch = mpatches.Patch(
    facecolor="white", edgecolor="grey", label="Plymouth LA boundary"
)

hn_zone_patch = mpatches.Patch(
    facecolor="none", edgecolor="#1f77b4", label="Existing heat network zones"
)

spatial_signature_patch = mpatches.Patch(
    facecolor="none", edgecolor="#800080", label="Selected spatial signatures"
)

# Combine auto and manual labels
handles, labels = plymouth_boundary.get_legend_handles_labels()

new_handles = handles + [boundary_patch, hn_zone_patch, spatial_signature_patch]
new_labels = labels + [
    "Plymouth LA boundary",
    "Existing heat network zones",
    "Selected spatial signatures",
]

plt.legend(
    handles=new_handles,
    labels=new_labels,
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    title="Layers",
)

plt.show()

# %% [markdown]
# Labelling UPRNs with existing HNZ labels **and** spatial signature types

# %%
plymouth_uprns_hnz_labelled_gdf = uprns.generate_gdf_uprn_coords(
    plymouth_uprns_hnz_labelled_df
)


# Test set of signature types
city_centre_types = [
    "Hyper concentrated urbanity",
    "Concentrated urbanity",
    "Metropolitan urbanity",
    "Regional urbanity",
    "Local urbanity",
    "Dense urban neighbourhoods",
]

plymouth_uprns_hnz_spatial_signature_labelled_df = (
    city_centres.label_gdf_city_centre_spatial_signatures_uprns(
        uprn_gdf=plymouth_uprns_hnz_labelled_gdf,
        spatial_signatures_gdf=spatial_signatures_gb_simplified_gdf,
        types=city_centre_types,
    )
)

# %% [markdown]
# **Comparison of zones by UPRNs coverage**

# %%
plymouth_uprns_hnz_spatial_signature_labelled_df.group_by(
    ["in_hn_zone", "in_city_centre"]
).len().with_columns((pl.col("len") / pl.col("len").sum()).alias("percent"))

# %%
# UPRNs in existing HNZ
hnz_true_df = plymouth_uprns_hnz_spatial_signature_labelled_df.filter(
    (pl.col("in_hn_zone") == True)
)
prop_true = hnz_true_df["in_city_centre"].mean()

print(
    f"Proportion of UPRNs in an existing heat network zone which are classified to be in a city centre (by the set of spatial signature types) [recall]: {prop_true:.3f}"
)

# %%
city_centre_true_df = plymouth_uprns_hnz_spatial_signature_labelled_df.filter(
    pl.col("in_city_centre") == True
)

prop_true = city_centre_true_df["in_hn_zone"].mean()

print(
    f"Proportion of UPRNs in a city centre area (as classified by the set of spatial signature types) which are already in an existing heat network zone [precision]: {prop_true:.3f}"
)

# %% [markdown]
# 05/12/25: City centres as a potential proxy performs better at capturing properties in existing HNZ (i.e. UPRN coverage), compared to spatial coverage. Mainly because of the higher density of properties in the selected urban area types.
# - 84% of UPRNs in existing HNZ are in city centre spatial signature types -> most properties already served are captured, showing our proxy effectively targets relevant properties
# - 82% of UPRNs in city centre types are in existing HNZ -> few properties outside existing HNZ are incorrectly flagged, indicating the proxy is precise
#
#
# For our purpose of flagging potential properties for future heat networks, city centre spatial signatures seem to be a suitable proxy as it captures the most relevant properties with moderately high precision, even if many of the existing HNZ total land area lies outside city centres.
