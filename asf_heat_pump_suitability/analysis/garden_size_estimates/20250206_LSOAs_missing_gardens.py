# %% [markdown]
# ## Investigate missing gardens
#
# There are some LSOAs in the final suitability results containing no properties with any garden size estimates. Here we investigate why.
#
# Possible reasons:
# - UPRNs don't have valid lat/lon
# - land extent parcels do not cover these LSOAs
# - building footprints do not cover these LSOAs

# %%
import polars as pl
import pandas as pd
import geopandas as gpd
import numpy as np
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.pipeline.prepare_features import (
    building_footprint,
    garden_size,
    garden_space_avg,
    land_extent,
    lat_lon,
)

# %%
# Load EPC data in final suitability dataset with garden size estimates
epc_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250206_2023_Q4_heat_pump_suitability_per_property.parquet",
    columns=["UPRN", "lsoa", "msoa", "garden_area_m2"],
)

# %% [markdown]
# ### Get LSOAs with no garden size data

# %%
# Get LSOAs with no garden size data
lsoa_df = epc_df.group_by("lsoa").agg(
    pl.col("garden_area_m2").mean().alias("mean_garden_area_m2"), pl.col("msoa").first()
)

null_df = lsoa_df.filter(pl.col("mean_garden_area_m2").is_null())
lsoa_names_df = pl.read_csv(
    config["data_source"]["EW_ons_lsoa_lad_lookup"], columns=["LSOA21CD", "LSOA21NM"]
)

# Add LSOA names
null_df = null_df.join(
    lsoa_names_df, left_on="lsoa", right_on="LSOA21CD", how="left"
).rename({"LSOA21NM": "lsoa_name"})

# Get LSOA name and code, and MSOA code of LSOAs with no garden size
null_lsoas = null_df["lsoa"].unique().to_list()
null_lsoa_names = null_df["lsoa_name"].unique().to_list()
null_msoas = null_df["msoa"].unique().to_list()

# %%
print(f"{len(null_lsoas)} LSOAs have no garden size data:")
print(null_lsoa_names)

# Remove LSOA code to get area name
area_names = set([lsoa[:-5] for lsoa in null_lsoa_names])
print(f"\nThese LSOAs are found in {len(area_names)} areas:")
print(area_names)

# %%
# Check that all properties in these LSOAs have no garden size
assert epc_df.filter(pl.col("lsoa").is_in(null_lsoas))[
    "garden_area_m2"
].is_null().sum() == len(epc_df.filter(pl.col("lsoa").is_in(null_lsoas)))

# %% [markdown]
# ### Confirm there is no MSOA-level average garden size data for these LSOAs:

# %%
# Load MSOA-level average garden size data
garden_space_avg_msoa_df = garden_space_avg.generate_df_garden_space_avg()

# %%
# We see there is no MSOA data for the LSOAs with null garden size values
garden_space_avg_msoa_df.filter(pl.col("MSOA code").is_in(null_msoas))

# %% [markdown]
# ### Confirm most UPRNs have lat / lon data:

# %%
# Load EPC with added features
raw_features_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/features/20250204_2023_Q4_EPC_features.parquet",
    columns=[
        "UPRN",
        "lsoa",
        "X_COORDINATE",
        "Y_COORDINATE",
        "LATITUDE",
        "LONGITUDE",
    ],
)

# %%
# Filter to properties in LSOAs with no garden size data
features_df = raw_features_df.filter(pl.col("lsoa").is_in(null_lsoas))
features_df = features_df.join(
    lsoa_names_df, left_on="lsoa", right_on="LSOA21CD", how="left"
).rename({"LSOA21NM": "lsoa_name"})

# %%
features_df.shape

# %%
print(
    f"{round(features_df['X_COORDINATE'].is_not_null().sum() / len(features_df) * 100, 2)}% of properties in these LSOAs have coordinate data"
)

# %% [markdown]
# ### Confirm UPRNs are covered by land extent files:

# %%
# Create point geoms from coordinates
features_gdf = lat_lon.generate_gdf_uprn_coords(features_df)

# %%
# Get land extent files covering the LSOAs with no garden size data
land_files = base_getters.list_obj_s3_location(
    "s3://asf-heat-pump-suitability/source_data/inspire_ew/"
)
area_names = {name.replace(" ", "_") for name in area_names}
land_files = {file for file in land_files if any(name in file for name in area_names)}

# %%
# Load land parcels for missing LSOAs
land_extent_gdf = []
for file in land_files:
    _gdf = land_extent.transform_gdf_land_parcels(f"s3://{file}")
    _gdf["land_file"] = file
    land_extent_gdf.append(_gdf)

land_extent_gdf = pd.concat(land_extent_gdf)

# %%
len(features_gdf)

# %%
print(
    f"{round(features_gdf.sjoin(land_extent_gdf, how='inner', predicate='intersects')['UPRN'].nunique() / features_gdf['UPRN'].nunique() * 100, 2)}% of properties with missing gardens are successfully joined to a land parcel"
)

# %% [markdown]
# ### Check UPRNs covered by building footprint files

# %%
microsoft_file_bounds = building_footprint.transform_df_uk_dataset_links()

# %%
# Load building footprint files which cover missing LSOAs
building_footprint_files = features_gdf.sjoin(
    microsoft_file_bounds, how="inner", predicate="intersects"
)["ms_url"].unique()

building_footprints_gdf = []
for file in building_footprint_files:
    _gdf = building_footprint.transform_gdf_building_footprints(file)
    _gdf["building_file"] = file
    building_footprints_gdf.append(_gdf)

building_footprints_gdf = pd.concat(building_footprints_gdf)

# %%
print(
    f"{round(features_gdf.sjoin(building_footprints_gdf, how='inner', predicate='intersects')['UPRN'].nunique() / features_gdf['UPRN'].nunique() * 100, 2)}% of properties with missing garden size get matched to a building footprint"
)

# %%
print("Equivalent to:")
print(
    features_gdf.sjoin(building_footprints_gdf, how="inner", predicate="intersects")[
        "UPRN"
    ].nunique()
)
print("properties")

# %% [markdown]
# ### Check if UPRNs get matched to land parcel + building pairs
#
# We can see from the above cells that most properties in this subset do not get matched to building footprints. However, some do, but this is contrary to what we see in the outputs of the pipeline, where none of the properties in this subset are matched to a garden. Let's check if they get joined to gardens or just building footprints.

# %%
# Get gdf of land parcels which match to a building footprint for the area covering the missing LSOAs
gardens_gdf = land_extent_gdf.sjoin(
    building_footprints_gdf, how="inner", predicate="intersects"
).drop(columns=["index_right"])

# %%
matches = features_gdf.sjoin(gardens_gdf, how="inner", predicate="intersects")
print(
    f"{round(matches['UPRN'].nunique() / features_gdf['UPRN'].nunique() * 100, 2)}% of properties with missing LSOAs are matched to a land parcel with buildings"
)

print("\nThey are located in the following LSOAs:")
print(matches["lsoa_name"].unique())

# %% [markdown]
# ### Check if land extent/building footprint file pairs for these LSOAs are missing from the file matching
#
# Since there are gardens that can be calculated for some of these properties with missing garden size, perhaps the land extent - building footprint file pairs are missing from the file matches.

# %%
# Get land extent and building footprint files for missing garden size estimates
missing_file_matches = matches[["land_file", "building_file"]].drop_duplicates()

# %%
# Recreate file matches used in pipeline
# Load land file bounds
land_file_bounds = gpd.read_file(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/inspire_file_bounds_EWS.geojson"
)

# Match land extent files with overlapping building footprint files
file_matches = garden_size.match_series_files_land_building(
    land_files_gdf=land_file_bounds, building_files_gdf=microsoft_file_bounds
)
file_matches_df = file_matches.reset_index().rename(columns={0: "building_file"})

# %%
# Check all matches are present in file_matches
exist_df = missing_file_matches.merge(
    file_matches_df,
    how="left",
    left_on=["land_file", "building_file"],
    right_on=["inspire_file_name", "building_file"],
    indicator="exist",
)
assert len(missing_file_matches) == len(exist_df)

# %% [markdown]
# ### Check that pipeline methodology works for these properties
#
# The file matches our present. Next we check that the functions we use in our pipeline work to generate these gardens.

# %%
# Get intersection of building footprint polygons and land polygons
intersection_gdf = garden_size.generate_gdf_land_building_overlay(
    land_parcels_gdf=land_extent_gdf,
    building_footprints_gdf=building_footprints_gdf,
)

# Get garden size
gardens_gdf = garden_size.generate_gdf_garden_size(intersection_gdf, land_extent_gdf)

# Match EPC UPRNs with land parcels and gardens using UPRN coordinates
# This will keep only EPC records for which garden size can be estimated
results_df = gpd.sjoin(
    features_gdf,
    gardens_gdf,
    how="inner",
    predicate="intersects",
).drop(columns=["geometry", "index_right"])

# %%
print(
    f"{round(results_df['UPRN'].nunique() / features_gdf['UPRN'].nunique() *100, 2)}% of properties missing a garden size can actually have a garden size calculated for them"
)

# %%
print("LSOAs still missing garden size due to absence of building footprint data:")
print(
    set(features_gdf["lsoa_name"].unique()).difference(
        set(results_df["lsoa_name"].unique())
    )
)

# %% [markdown]
# ### Conclusion
#
# It appears that our results are missing some garden size estimates that should be present.
#
# As of 11 Feb 2025, there has been one update (3 Feb 2025) to the Microsoft Building footprint data where new building footprints were added since running the pipeline that produced the garden size estimates used in this notebook. It's possible that those properties missing garden size estimates, which we can now fill, had building footprints added during the 3 Feb 2025 update.
#
# It's also possible that during crashing and restarting of the pipeline, there was a land extent - building footprint file pairing that was accidentally missed.
#
# As for the remaining LSOAs which have no garden size estimates, this appears to be due to absence of building footprint data. When looking at the map of coverage of Microsoft Global Building Footprints (URL: https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/images/country-overview.png), we can see that the south western tip of England appears to be missing coverage. It's possible that explains the absence of data in Plymouth. (Note I'm unsure if this map is up to date with their data releases.)
#
# The other LSOAs are all in London. I don't have a theory as to why they are missing building footprints, but it appears to be the case that they are genuinely missing both building footprint and MSOA-level average garden size data.

# %%
