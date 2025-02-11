# %% [markdown]
# ## Building conservation area data
#
# We have a dataset of building conservation areas from Historic England and Welsh Gov, covering Local Authority Districts (LADs) in England and Wales. Some LADs are missing building conservation area data. We want to assess whether we can make the assumption that if an LAD has ANY building conservation area data, then it can be assumed to be complete. We will test this assumption by looking at a few example areas.

# %%
import polars as pl
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import (
    epc,
    protected_areas,
    lat_lon,
)

# %% [markdown]
# ### Load and transform datasets

# %%
# Load England and Wales building conservation area data and concatenate
e_gdf = (
    get_datasets.load_gdf_historic_england_conservation_areas(
        columns=["name", "geometry"]
    )
    .to_crs("EPSG:27700")
    .rename(columns={"name": "sitename"})
)
w_gdf = get_datasets.load_gdf_welsh_gov_conservation_areas(
    columns=["sitename", "geometry"]
)
full_cons_areas_gdf = pd.concat([e_gdf, w_gdf]).drop_duplicates(subset=["geometry"])
full_cons_areas_gdf["in_conservation_area_ew"] = True

# %%
# Load geospatial boundaries of local authorities
council_bounds = get_datasets.load_gdf_ons_council_bounds()

# %%
full_cons_areas_gdf["cons_area_size_m2"] = full_cons_areas_gdf["geometry"].area
full_cons_areas_gdf = gpd.overlay(
    full_cons_areas_gdf, council_bounds, how="intersection", keep_geom_type=False
)
full_cons_areas_gdf["overlay_size_m2"] = full_cons_areas_gdf["geometry"].area
full_cons_areas_gdf["overlay_pc"] = (
    full_cons_areas_gdf["overlay_size_m2"]
    / full_cons_areas_gdf["cons_area_size_m2"]
    * 100
)
cons_areas_gdf = full_cons_areas_gdf[full_cons_areas_gdf["overlay_pc"] > 10].copy()

# %%
# Join conservation areas to their local authorities using geospatial join
cons_areas_gdf = cons_areas_gdf.groupby("LAD23CD").agg(
    {
        "in_conservation_area_ew": "count",
        "LAD23NM": "first",
        "LAD23CD": "first",
        "sitename": list,
    }
)

cons_areas_gdf["lad_conservation_area_data_available_ew"] = cons_areas_gdf[
    "in_conservation_area_ew"
].astype(bool)
cons_areas_df = pl.from_pandas(cons_areas_gdf)

# %%
# Load manually collated building conservation area count data for 51 local authorities and join to conservation areas
counts_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/evaluation/building_conservation_area_datasets/building_conservation_area_counts_sample.csv"
)
cons_areas_df = cons_areas_df.join(counts_df, how="inner", on="LAD23NM")

# %% [markdown]
# ### Analysis

# %%
# Add columns to indicate if our dataset has too many/too few conservation areas compared to that reported by local authorities
cons_areas_df = cons_areas_df.with_columns(
    pl.when(pl.col("in_conservation_area_ew") == pl.col("conservation_area_count"))
    .then(pl.lit("same"))
    .when(pl.col("in_conservation_area_ew") > pl.col("conservation_area_count"))
    .then(pl.lit("too many"))
    .when(pl.col("in_conservation_area_ew") < pl.col("conservation_area_count"))
    .then(pl.lit("not enough"))
    .alias("full_dataset_vs_councils"),
    (pl.col("in_conservation_area_ew") - pl.col("conservation_area_count")).alias(
        "diff"
    ),
)

# %%
cons_areas_df["full_dataset_vs_councils"].value_counts(normalize=True)

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

for c, ax in zip(["not enough", "too many"], axes.ravel()):
    _df = cons_areas_df.filter(pl.col("full_dataset_vs_councils") == c)
    print(c)
    print(_df["diff"].describe())
    ax.hist(_df["diff"])
    ax.set_title(c.capitalize() + " in Historic England dataset")
    ax.set_xlabel("Difference")
    ax.set_ylabel("Count of Local Authorities")

fig.suptitle(
    "Distribution of differences between number of building\nconservation areas in Historic England vs Local Authority datasets"
)
fig.tight_layout()

# %% [markdown]
# ### Compare Historic England and Wales Gov polygons to local authority polygons

# %%
cons_areas_df = epc.extend_df_country_col(cons_areas_df, lsoa_col="LAD23CD")

# %%
# LADs with far too many / few conservation areas
sample_lads = cons_areas_df.filter((pl.col("diff") <= -30) | (pl.col("diff") >= 10))[
    "LAD23CD"
].to_list()

# Add a random sample of LADs from England and Wales
sample_lads.extend(
    cons_areas_df.filter(pl.col("full_dataset_vs_councils") != "same")
    .group_by(["country", "full_dataset_vs_councils"])
    .agg(pl.all().sample(1, with_replacement=False, seed=4))
    .explode(pl.all().exclude(["country", "full_dataset_vs_councils"]))["LAD23CD"]
    .to_list()
)

# %%
sample_df = cons_areas_df.filter(pl.col("LAD23CD").is_in(sample_lads))
sample_df

# %% [markdown]
# ### East Hampshire building conservation areas
#
# From the sample above, the only Local Authority that seems to publish polygon data on their website is East Hampshire (source: https://www.easthants.gov.uk/open-data). The Historic England dataset has too many conservation areas compared to the East Hampshire Local Authority's own dataset.

# %%
# Load East Hampshire building conservation areas and drop duplicate geometries
eh_cons_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/evaluation/building_conservation_area_datasets/east_hampshire_conservation_areas.csv"
)
eh_cons_gdf = eh_cons_gdf.drop_duplicates(subset=["geometry"])

# %%
# The Historic England dataset lists 57 building conservation areas for East Hampshire
# East Hampshire's website lists 43, however we can see 56 in the dataset
# We can see it's because there appear to be duplicate site names (which we assumed to be the same site) but with different geometries
# This seems to be due to changes in the geometry (extensions/ deletions)
# However, the changes are not dated so it's not clear how to determine which is the most current geometry
eh_cons_gdf.shape

# %%
# Dropping duplicate names gives us 45 sites. This is still 2 more than the 43 reported on their website.
eh_cons_gdf.drop_duplicates(subset=["name"]).shape

# %%
# Historic England also has duplicate names. So after removing them, we have 44 building conservation areas, 1 less than reported by East Hampshire
len(set(cons_areas_df.filter(pl.col("LAD23NM") == "East Hampshire")["sitename"][0]))

# %%
# Check which conservation areas are missing from Historic England
set(eh_cons_gdf.drop_duplicates(subset=["name"])["name"]).difference(
    set(cons_areas_df.filter(pl.col("LAD23NM") == "East Hampshire")["sitename"][0])
)

# %%
# We can see that 'Sir George Staunton' building conservation area was joined to East Hampshire, it just has a low % overlay (23%) so was filtered out.
full_cons_areas_gdf[full_cons_areas_gdf["sitename"] == "Sir George Staunton"]

# %% [markdown]
# ### Plymouth building conservation areas
#
# We also have Plymouth data (source: https://plymouth.thedata.place/dataset/7d4db8c4-6ac3-4e13-957b-4745db3b357b/resource/e474cdba-bbf4-41c0-bc14-894c81f69f1e/download/conservation_area_plymouth.geojson). The Historic England dataset has the same number of building conservation areas as the Plymouth dataset and they are all the same sites, as shown below.

# %%
plymouth_cons_gdf = gpd.read_file(
    "https://plymouth.thedata.place/dataset/7d4db8c4-6ac3-4e13-957b-4745db3b357b/resource/e474cdba-bbf4-41c0-bc14-894c81f69f1e/download/conservation_area_plymouth.geojson"
)
plymouth_cons_gdf = plymouth_cons_gdf.drop_duplicates(subset=["geometry"])

# %%
# According to Plymouth Local Authority website, there are 15 building conservation areas in Plymouth
# Historic England also has 15
plymouth_cons_gdf.shape

# %%
# Building conservation areas in Plymouth from Historic England
set(cons_areas_df.filter(pl.col("LAD23NM") == "Plymouth")["sitename"][0])

# %%
# Check that the building conservation areas are the same in Historic England and Plymouth datasets
set(plymouth_cons_gdf.drop_duplicates(subset=["name"])["name"]).difference(
    set(set(cons_areas_df.filter(pl.col("LAD23NM") == "Plymouth")["sitename"][0]))
)

# %% [markdown]
# ### Check if we can deduplicate sites

# %%
# Load England building conservation areas from Historic England
e_gdf = (
    get_datasets.load_gdf_historic_england_conservation_areas()
    .drop_duplicates(subset="geometry")
    .to_crs("EPSG:27700")
)

# %%
print("Number of duplicate site names:")
print(len(e_gdf[e_gdf["name"].duplicated()]))

# %%
# Get duplicated sites.
# It looks like `entry-date` is variable across duplicates.
# We will test deduplicating by dropping all but the latest `entry-date` and see if that improves our results
duplicates_df = e_gdf[e_gdf["name"].duplicated(keep=False)].sort_values(by="name")
duplicates_df

# %% [markdown]
# ### Analysis on deduplicated data

# %%
# Load England and Wales building conservation area data and concatenate
# Removing duplicates by keeping latest `entry-date` does to results
e_gdf = (
    e_gdf.sort_values(by="entry-date", ascending=False)
    .drop_duplicates(subset="name", keep="first")[["name", "geometry"]]
    .rename(columns={"name": "sitename"})
)

# %%
# Below we run the same as above
w_gdf = get_datasets.load_gdf_welsh_gov_conservation_areas(
    columns=["sitename", "geometry"]
)
full_cons_areas_gdf = pd.concat([e_gdf, w_gdf]).drop_duplicates(subset=["geometry"])
full_cons_areas_gdf["in_conservation_area_ew"] = True

# Load geospatial boundaries of local authorities
council_bounds = get_datasets.load_gdf_ons_council_bounds()

full_cons_areas_gdf["cons_area_size_m2"] = full_cons_areas_gdf["geometry"].area
full_cons_areas_gdf = gpd.overlay(
    full_cons_areas_gdf, council_bounds, how="intersection", keep_geom_type=False
)
full_cons_areas_gdf["overlay_size_m2"] = full_cons_areas_gdf["geometry"].area
full_cons_areas_gdf["overlay_pc"] = (
    full_cons_areas_gdf["overlay_size_m2"]
    / full_cons_areas_gdf["cons_area_size_m2"]
    * 100
)
cons_areas_gdf = full_cons_areas_gdf[full_cons_areas_gdf["overlay_pc"] > 90].copy()

# %%
# Join conservation areas to their local authorities using geospatial join
cons_areas_gdf = cons_areas_gdf.groupby("LAD23CD").agg(
    {
        "in_conservation_area_ew": "count",
        "LAD23NM": "first",
        "LAD23CD": "first",
        "sitename": list,
    }
)

cons_areas_gdf["lad_conservation_area_data_available_ew"] = cons_areas_gdf[
    "in_conservation_area_ew"
].astype(bool)
cons_areas_df = pl.from_pandas(cons_areas_gdf)

# Join manually collated building conservation area count data for 50 local authorities and join to conservation areas
cons_areas_df = cons_areas_df.join(counts_df, how="inner", on="LAD23NM")

# %%
# Add columns to indicate if our dataset has too many/too few conservation areas compared to that reported by local authorities
cons_areas_df = cons_areas_df.with_columns(
    pl.when(pl.col("in_conservation_area_ew") == pl.col("conservation_area_count"))
    .then(pl.lit("same"))
    .when(pl.col("in_conservation_area_ew") > pl.col("conservation_area_count"))
    .then(pl.lit("too many"))
    .when(pl.col("in_conservation_area_ew") < pl.col("conservation_area_count"))
    .then(pl.lit("not enough"))
    .alias("full_dataset_vs_councils"),
    (pl.col("in_conservation_area_ew") - pl.col("conservation_area_count")).alias(
        "diff"
    ),
)

cons_areas_df["full_dataset_vs_councils"].value_counts(normalize=True)

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

for c, ax in zip(["not enough", "too many"], axes.ravel()):
    _df = cons_areas_df.filter(pl.col("full_dataset_vs_councils") == c)
    print(c)
    print(_df["diff"].describe())
    ax.hist(_df["diff"])
    ax.set_title(c.capitalize() + " in Historic England dataset")
    ax.set_xlabel("Difference")
    ax.set_ylabel("Count of Local Authorities")

fig.suptitle(
    "Distribution of differences between number of building\nconservation areas in Historic England vs Local Authority datasets (after deduplication of site names)"
)
fig.tight_layout()

# %% [markdown]
# It seems to make our results worse for the test dataset. It looks like the result is that we remove too many conservation areas so lots of Local Authorities now have not enough. This seems strange considering that duplicate names should represent the same site and therefore get matched with the same Local Authority.
#
# Potential reasons:
# 1. There are different conservation areas that have the same name.
# 2. Historic England aggregates Local Authority level data. Where a conservation area overlaps 2 or more Local Authorities, perhaps each local authority publishes only partial polygons of the part that overlap their area.
# 3. There are multiple different polygons which have the same site name but represent different parts of the same site.
