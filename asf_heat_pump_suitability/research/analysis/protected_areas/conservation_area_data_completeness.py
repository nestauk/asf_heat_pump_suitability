# %% [markdown]
# ## Building conservation area data
#
# We have a dataset of building conservation areas from Historic England and Welsh Gov, covering Local Authority Districts (LADs) in England and Wales. Some LADs are missing building conservation area data. We want to assess whether we can make the assumption that if an LAD has ANY building conservation area data, then it can be assumed to be complete. We will test this assumption by looking at a few example areas.
#
# The purpose of this analysis is ultimately to inform our `in_protected_area` feature in our Heat Pump Suitability features dataset. We identify properties in building conservation areas by joining their geospatial coordinates with conservation area polygons. If a property is NOT in a conservation area, it can either be labelled as `in_protected_area == False` or as `null`. The results of this analysis will inform us whether we feel confident enough to assign a `False` label to properties within LADs which have ANY conservation area polygons, or whether those properties should be labelled as `null`.
#
# For this analysis, we have two key datasets:
# 1. Geospatial polygons of building conservation areas from Historic England and the Welsh government. This is the dataset we use to create the `in_protected_area` feature. We are aiming to verify the coverage of conservation areas for each local authority.
# 2. A manually collated building conservation area count dataset for 51 local authorities. We visited the gov websites of these 51 local authorities and recorded the number of conservation areas they report within their boundaries. This provides the ground truth to compare our counts against.

# %%
import polars as pl
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from asf_heat_pump_suitability.getters import load_data, load_geodata
from asf_heat_pump_suitability.pipeline.transform import protected_areas

# %% [markdown]
# ### Load and transform datasets

# %%
# Load England and Wales building conservation area data and concatenate
e_gdf = (
    load_geodata.load_gdf_historic_england_conservation_areas(
        columns=["name", "geometry"]
    )
    .to_crs("EPSG:27700")
    .rename(columns={"name": "sitename"})
)
w_gdf = load_geodata.load_gdf_welsh_gov_conservation_areas(
    columns=["sitename", "geometry"]
)
full_cons_areas_gdf = pd.concat([e_gdf, w_gdf]).drop_duplicates(subset=["geometry"])
full_cons_areas_gdf["in_conservation_area_ew"] = True

# %%
# Load geospatial boundaries of local authorities
council_bounds = load_data.load_gdf_ons_council_bounds()

# %% [markdown]
# Because this analysis is focussing on conservation area data availability for Local Authority Districts (councils), the steps below are meant to join the conservation areas to their local authorities. I tried out a couple of different join methods and this one seemed to produce the best results when comparing to our count data.
#
# sjoin with 'intersects' predicate seemed to be too permissive and there was a high number of LADs with too many conservation areas. sjoin with 'contains' predicate (as in, LAD contains conservation areas) seemed to be too restrictive and there was a high number of LADs with not enough conservation areas.
#
# As a result, I chose this methodology where we calculate what percentage of each conservation area fell within the LAD, and then kept those where at least 10% of the conservation area was found in the LAD OR the size of conservation area within the LAD was at least 50m2. I chose 50m2 because I think it's roughly the size of a small building footprint plus garden so in theory, at least one property could be in the conservation area. This was to avoid small intersections with neighbouring LAD conservation areas being matched. My assumption is that a very small intersection between a LAD and conservation area wouldn't be considered in the conservation area count by the LAD.
#
# 9 conservation areas are lost during this process. They are dropped after the initial overlay function so there must not be any LAD boundary data for the areas where they are located.

# %%
# Create unique ID for each building conservation area
full_cons_areas_gdf["site_id"] = [i for i in range(0, len(full_cons_areas_gdf))]
print("Number of unique conservation areas before matching to LAD bounds:")
print(full_cons_areas_gdf["site_id"].nunique())
all_cons_area_ids = set(full_cons_areas_gdf["site_id"])

# Get area of overlay between all LADs and conservation areas
overlay_cons_areas_gdf = full_cons_areas_gdf.copy()
overlay_cons_areas_gdf["cons_area_size_m2"] = overlay_cons_areas_gdf["geometry"].area
overlay_cons_areas_gdf = gpd.overlay(
    overlay_cons_areas_gdf, council_bounds, how="intersection", keep_geom_type=False
)
overlay_cons_areas_gdf["overlay_size_m2"] = overlay_cons_areas_gdf["geometry"].area

# Calculate percentage of conservation area which is inside the LAD
overlay_cons_areas_gdf["overlay_pc"] = (
    overlay_cons_areas_gdf["overlay_size_m2"]
    / overlay_cons_areas_gdf["cons_area_size_m2"]
    * 100
)

# Filter to conservation area-to-LAD matches where at least 50m2 or 10% of the conservation area is inside the LAD
cons_areas_gdf = overlay_cons_areas_gdf[
    (overlay_cons_areas_gdf["overlay_size_m2"] >= 50)
    | (overlay_cons_areas_gdf["overlay_pc"] >= 10)
].copy()
print("\nNumber of unique conservation areas after matching to LAD bounds:")
print(cons_areas_gdf["site_id"].nunique())
remaining_cons_area_ids = set(cons_areas_gdf["site_id"])

# %%
cons_areas_gdf = overlay_cons_areas_gdf[
    (overlay_cons_areas_gdf["overlay_size_m2"] >= 50)
    | (overlay_cons_areas_gdf["overlay_pc"] >= 10)
].copy()

# %%
cons_areas_gdf

# %%
# Drop duplicate conservation area site IDs in the same LAD so we get an accurate count
cons_areas_df = cons_areas_gdf.drop_duplicates(subset=["LAD23CD", "site_id"])

# Get conservation area count per LAD
cons_areas_df = cons_areas_df.groupby("LAD23CD").agg(
    {
        "in_conservation_area_ew": "count",
        "LAD23NM": "first",
        "LAD23CD": "first",
        "sitename": list,
    }
)

cons_areas_df = pl.from_pandas(cons_areas_df)

# %%
# Load manually collated building conservation area count data for 51 local authorities and join to conservation areas
counts_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/evaluation/building_conservation_area_datasets/building_conservation_area_counts_sample.csv"
)
cons_areas_df = cons_areas_df.join(counts_df, how="inner", on="LAD23NM").rename(
    {
        "in_conservation_area_ew": "polygons_count",
        "conservation_area_count": "council_count",
    }
)

# %% [markdown]
# ### Analysis
#
# Here we compare the counts of conservation areas in LADs from the gov website sources, versus the geospatial dataset. This will inform us whether we can assume that if a LAD has at least one conservation area in the geospatial dataset, then we can assume (for the purpose of labelling properties with a protected area flag) that the conservation polygons for that LAD is complete.
#
# We can see that the majority of LADs have the same number or more conservation area polygons in the geospatial dataset as reported on their gov websites. This suggests that the polygon dataset has good coverage. On average, there are about 7 areas too many for those LADs with excess polygons.
#
# For the 18% of LADs with fewer conservation area polygons than the website reports, the average is 26 areas too few.

# %%
# Add columns to indicate if our dataset has too many/too few conservation areas compared to that reported by local authorities
cons_areas_df = cons_areas_df.with_columns(
    pl.when(pl.col("polygons_count") == pl.col("council_count"))
    .then(pl.lit("same"))
    .when(pl.col("polygons_count") > pl.col("council_count"))
    .then(pl.lit("too many"))
    .when(pl.col("polygons_count") < pl.col("council_count"))
    .then(pl.lit("not enough"))
    .alias("polygons_vs_councils"),
    (pl.col("polygons_count") / pl.col("council_count") * 100).alias(
        "polygon_pc_of_total"
    ),
    (pl.col("polygons_count") - pl.col("council_count")).alias("diff"),
)

# %%
cons_areas_df["polygons_vs_councils"].value_counts(normalize=True)

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

for c, ax in zip(["not enough", "too many"], axes.ravel()):
    _df = cons_areas_df.filter(pl.col("polygons_vs_councils") == c)
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

# %%
plt.boxplot(cons_areas_df["polygon_pc_of_total"])
plt.title(
    "Percentage of conservation areas reported by LADs which are present in HE polygon data"
    "\n>100% means there are more conservation polygons than reported conservation areas in LAD"
)
plt.ylabel("Percentage of LAD conservation areas")
plt.show()

# %% [markdown]
# ### Compare Historic England and Wales Gov polygons to local authority polygons
#
# Here we take a sample of LADs with far too many (10+) or too few (-10) conservation areas and a random sample from England and Wales of LADs with a different number of polygons in HE vs council data. We will attempt to find conservation area polygon data published by the local authorities themselves and then compare the Historic England polygons with the local authority polygons to identify why there are differences.

# %%
# Add a column to assign country
cons_areas_df = protected_areas.extend_df_country_col(cons_areas_df, lsoa_col="LAD23CD")

# LADs with far too many / few conservation areas
sample_lads = cons_areas_df.filter((pl.col("diff") <= -10) | (pl.col("diff") >= 10))[
    "LAD23CD"
].to_list()

# Add a random sample of LADs from England and Wales which have different numbers of polygons
sample_lads.extend(
    cons_areas_df.filter(pl.col("polygons_vs_councils") != "same")
    .group_by(["country", "polygons_vs_councils"])
    .agg(pl.all().sample(1, with_replacement=False, seed=4))
    .explode(pl.all().exclude(["country", "polygons_vs_councils"]))["LAD23CD"]
    .to_list()
)

sample_df = cons_areas_df.filter(pl.col("LAD23CD").is_in(sample_lads))
sample_df

# %%
sample_df["LAD23NM"].to_list()

# %% [markdown]
# ### East Hampshire building conservation areas
#
# From the sample above, the only Local Authority that seems to publish polygon data on their website is East Hampshire (source: https://www.easthants.gov.uk/open-data). The Historic England dataset has too many conservation areas compared to the East Hampshire Local Authority's own dataset. Here, we will compare the datasets.

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
# This seems to be due to changes in the geometry (extensions/ deletions as indicated by the 'notes' column)
# However, the changes are not dated so it's not clear how to determine which is the most current geometry
eh_cons_gdf.shape

# %%
# Dropping duplicate names gives us 45 sites. This is still 2 more than the 43 reported on their website.
eh_cons_gdf.drop_duplicates(subset=["name"]).shape

# %%
# Historic England also has duplicate names. So after removing them, we have 46 building conservation areas, 1 more than reported by East Hampshire
len(set(cons_areas_df.filter(pl.col("LAD23NM") == "East Hampshire")["sitename"][0]))

# %%
# Check which conservation areas are extra
set(
    cons_areas_df.filter(pl.col("LAD23NM") == "East Hampshire")["sitename"][0]
).difference(set(eh_cons_gdf.drop_duplicates(subset=["name"])["name"]))

# %%
# Wey Valley has a 7000m2 overlap with East Hampshire so seems like a reasonable match
overlay_cons_areas_gdf[overlay_cons_areas_gdf["sitename"] == "Wey Valley"]

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
# Check that the building conservation areas are the same in Historic England and Plymouth datasets
set(plymouth_cons_gdf.drop_duplicates(subset=["name"])["name"]).difference(
    set(set(cons_areas_df.filter(pl.col("LAD23NM") == "Plymouth")["sitename"][0]))
)

# %% [markdown]
# ### Conclusion
#
# After discussion in a sprint review, we determined that we have sufficient evidence to label properties as `in_protected_area == False` if a property is in a LAD which has at least some conservation area data, rather than labelling these as `null`. Although this will potentially mislabel some properties as `False` when they should be `null` (or `True` if there is a conservation area polygon missing from the geospatial dataset), we are comfortable that the majority of properties this applies to will be correctly labelled.
#
# The rules for assigning the `in_protected_area` label are as follows:
# - `True` if the property is in a conservation area polygon
# - `False` if the property has coordinate data, is not in a conservation area polygon, and is in a Local Authority that has at least one conservation area polygon
# - `Null` if the property has no coordinate data or is not in a conservation area polygon and is in a Local Authority that does not have any conservation area polygons
#
# SIDE NOTE: There are duplicate site names in the polygon dataset. They have different geometries so we won't deduplicate on site name, only geometry.
#
# Potential reasons for multiple geometries with the same name:
# 1. There are different conservation areas that have the same name.
# 2. Historic England aggregates Local Authority level data. Where a conservation area overlaps 2 or more Local Authorities, perhaps each local authority publishes only partial polygons of the part that overlap their area.
# 3. There are multiple different polygons which have the same site name but represent different parts of the same site.

# %% [markdown]
#
