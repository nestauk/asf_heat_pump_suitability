# %% [markdown]
# ## Explore creating a full dataset of all residential properties in Plymouth
#
# This notebook explores the methodology for creating a complete residential property dataset in Plymouth using only open-source data. The final dataset includes relevant property-level data to assess physical feasibility for ASHPs, SGLs, and HNs. The features included are:
# - in listed building
# - in building conservation area
# - outdoor space [to be added]
# - in existing / planned heat network zone
# - tenure
# - property type
# - heating fuel [to be added]
# - off gas status

# %% [markdown]
# ## Import libraries

# %%
import geopandas as gpd
import polars as pl
import pandas as pd
import fiona
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import (
    lat_lon,
    listed_buildings,
    protected_areas,
    off_gas,
    anchor_properties,
    output_areas,
)
from asf_heat_pump_suitability.analysis.flats_on_fossils.features import fuel_type
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_sample

# %% [markdown]
# ## Load data

# %%
fiona.listlayers("s3://asf-heat-pump-suitability/source_data/opmplc_gb.gpkg")

# %%
print("LOADING DATASETS TO GET RESIDENTIAL UPRNS FOR PLYMOUTH...")
print("Loading OS UPRN...")
os_uprn_df = get_datasets.get_df_osopen_uprn_latlon()

print("\nLoading LA boundaries...")
la_boundaries_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/source_data/Local_Authority_Districts_December_2023_Boundaries_UK_BFE_-2600600853110041429/LAD_DEC_2023_UK_BFE.shp"
)

print("\nLoading OS OpenMap Local - Buildings...")
os_openmap_building_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/OS OpenMap Local (ESRI Shape File) SX/data/SX_Building.shp"
)

print("\nLoading OS OpenMap Local - Important Buildings...")
os_openmap_importantbuilding_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/OS OpenMap Local (ESRI Shape File) SX/data/SX_ImportantBuilding.shp"
)

print("\nLoading OS OpenMap Local - Railways...")
os_openmap_railway_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/OS OpenMap Local (ESRI Shape File) SX/data/SX_RailwayStation.shp"
)

# --------------------------------------------------------------------------#

print("\nLOADING PROPERTY-LEVEL FEATURE DATASETS...")
print("\nLoading listed buildings...")
listed_buildings_gdf = gpd.read_file(
    "../spatial_clustering/data/National_Heritage_List_for_England_NHLE_v02_VIEW_-464524051049198649/Listed_Building_points.shp"
)

print("\nLoading building conservation areas...")
cons_areas_gdf = gpd.read_file(
    "../spatial_clustering/data/conservation-area (1).geojson"
)

print("\nLoading existing / planned HN zones...")
hn_zones_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/heat_network_desnz_data/heat-network-zone-map-Plymouth.gpkg",
    layer="heat-network-zone-map-Plymouth",
)

print("\nLoading outdoor space...")
outdoor_space_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/20250224_2023_Q4_EPC_garden_size_estimates_EWS_deduplicated.parquet"
)

print("\nLoading off gas postcodes...")
off_gas_list = off_gas.process_off_gas_data()

print("\nLoading EPC...")
epc_df = pl.read_parquet(
    "s3://asf-daps/lakehouse/2025_Q1/processed/epc/deduplicated/processed_dedupl-0.parquet",
    columns=[
        "UPRN",
        "POSTCODE",
        "PROPERTY_TYPE",
        "BUILT_FORM",
        "TENURE",
        "MAIN_FUEL",
        "MAINHEAT_DESCRIPTION",
    ],
)

print("\nAdding EPC property type...")
epc_df = prepare_sample.add_col_property_type(epc_df)

# -------------------------------------------------------------------------#

print("\nLOADING DATASETS TO FILL MISSING DATA")
print("\nLoading OS Code-Point Open...")
code_point_df = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/codepo_gb.gpkg",
    layers="codepoint",
)

# %% [markdown]
# ## Filter to Plymouth only

# %%
# Convert UPRN coordinates to geopoints
os_uprn_gdf = lat_lon.generate_gdf_uprn_coords(os_uprn_df)

# Get polygon of Plymouth LA
plymouth_polygon = la_boundaries_gdf[
    la_boundaries_gdf["LAD23NM"].str.contains("Plymouth")
]["geometry"].values[0]

# Filter UPRN and OpenMap datasets to Plymouth only
os_uprn_plymouth_gdf = os_uprn_gdf[os_uprn_gdf.within(plymouth_polygon)]
os_openmap_buildings_plymouth_gdf = os_openmap_building_gdf[
    os_openmap_building_gdf.within(plymouth_polygon)
]
os_openmap_importantbuildings_plymouth_gdf = os_openmap_importantbuilding_gdf[
    os_openmap_importantbuilding_gdf.within(plymouth_polygon)
]

# %% [markdown]
# ## Identify residential UPRNs and building footprints

# %%
# View types of building
os_openmap_importantbuildings_plymouth_gdf.groupby("BUILDGTHEM")[
    "CLASSIFICA"
].value_counts(dropna=False)

# %%
# Identify all UPRNs in a building - these contain residential addresses
uprns_in_buildings = os_openmap_buildings_plymouth_gdf.sjoin(
    os_uprn_plymouth_gdf, how="inner", predicate="contains"
)["UPRN"].tolist()

# Identify building types which are unlikely to contain residential dwellings
no_residential_overlap_building_types = [
    "Port Consisting of Docks and Nautical Berthing",
    "Fire Station",
    "Hospital",
    "Non State Secondary Education",
    "Secondary Education",
    "Higher or University Education",
    "Primary Education",
    "Post Office",
    "Place Of Worship",
    "Medical Care Accommodation",
    "Museum",
    "Special Needs Education",
    "Further Education",
    "Non State Primary Education",
    "Coach Station",
    "Police Station",
    "Sports And Leisure Centre",
    "Vehicular Ferry Terminal",
    "Hospice",
    "Bus Station",
]

# Create gdf of important buildings and railway stations which are unlikely to contain residential dwellings
exclude_buildings_plymouth_gdf = os_openmap_importantbuildings_plymouth_gdf[
    os_openmap_importantbuildings_plymouth_gdf["CLASSIFICA"].isin(
        no_residential_overlap_building_types
    )
]
railway_stations_gdf = os_openmap_buildings_plymouth_gdf.sjoin(
    os_openmap_railway_gdf, how="inner", predicate="contains"
)
exclude_buildings_plymouth_gdf = pd.concat(
    [exclude_buildings_plymouth_gdf[["geometry"]], railway_stations_gdf[["geometry"]]]
)

# Identify UPRNs which are in these buildings - i.e. UPRNs which are likely to represent non-residential addresses
exclude_uprns = os_uprn_plymouth_gdf.sjoin(
    exclude_buildings_plymouth_gdf, how="inner", predicate="within"
)["UPRN"].tolist()

# Filter UPRN dataset to only residential UPRNs - there will still be some commercial UPRNs from mixed-use buildings
plymouth_residential_uprns_gdf = os_uprn_plymouth_gdf[
    (~os_uprn_plymouth_gdf["UPRN"].isin(exclude_uprns))
    & (os_uprn_plymouth_gdf["UPRN"].isin(uprns_in_buildings))
]

# Filter building footprints to only those which contain residential UPRNs
plymouth_residential_buildings_gdf = os_openmap_buildings_plymouth_gdf.sjoin(
    plymouth_residential_uprns_gdf, how="inner", predicate="contains"
).drop_duplicates(subset=["ID"])
plymouth_residential_buildings_gdf["building_area_m2"] = (
    plymouth_residential_buildings_gdf.area
)

# %%
plymouth_residential_buildings_gdf.plot()

# %%
plymouth_residential_uprns_gdf.shape

# %% [markdown]
# ## Add features
#
# - in listed building
# - in building conservation area
# - outdoor space
# - in existing / planned heat network zone
# - tenure
# - property type
# - heating fuel
# - off gas status

# %%
# Identify UPRNs in listed buildings
listed_building_uprns = (
    plymouth_residential_uprns_gdf.sjoin_nearest(
        listed_buildings_gdf, how="inner", max_distance=10
    )["UPRN"]
    .unique()
    .tolist()
)

# UPRNs in building conservation areas
cons_area_uprns = plymouth_residential_uprns_gdf.sjoin(
    cons_areas_gdf.to_crs(epsg=27700), how="inner", predicate="intersects"
)["UPRN"].tolist()

# Reformat UPRN for join
outdoor_space_df = outdoor_space_df.with_columns(
    pl.col("UPRN").str.replace(".0", "", literal=True).cast(pl.Int64).alias("UPRN")
)

# UPRNs in HNs
uprns_in_hn = plymouth_residential_uprns_gdf.sjoin(
    hn_zones_gdf.drop(columns=["index_right"]), predicate="within"
)["UPRN"].tolist()

# %%
len(listed_building_uprns) / len(plymouth_residential_uprns_gdf) * 100

# %%
# Standardise postcode column - remove space and set to upper case
epc_df = output_areas.standardise_col_postcode(epc_df, pcd_col="POSTCODE")
epc_df = epc_df.with_columns(
    # Standardise UPRN
    pl.col("UPRN")
    .str.replace(".0", "", literal=True)
    .cast(pl.Int64, strict=False)
    .alias("UPRN"),
    pl.col("MAIN_FUEL").cast(pl.String).alias("MAIN_FUEL"),
    pl.col("PROPERTY_TYPE").cast(pl.String).alias("PROPERTY_TYPE"),
)

# Process central heating fuel type information
epc_df = fuel_type.extend_df_fuel_type(epc_df, epc_col="MAIN_FUEL", name="fuel_type")
epc_df = fuel_type.extend_df_fuel_type(
    epc_df, epc_col="MAINHEAT_DESCRIPTION", name="fill_fuel_type"
)
epc_df = epc_df.with_columns(
    pl.col("fuel_type").fill_null(pl.col("fill_fuel_type"))
).drop(["fill_fuel_type"])

# %%
# Create features dataset with initial property-level features
features_df = (
    pl.from_pandas(plymouth_residential_uprns_gdf[["UPRN"]])
    .with_columns(
        pl.when(pl.col("UPRN").is_in(listed_building_uprns))
        .then(True)
        .otherwise(False)
        .alias("in_listed_building"),
        pl.when(pl.col("UPRN").is_in(cons_area_uprns))
        .then(True)
        .otherwise(False)
        .alias("in_cons_area"),
        pl.when(pl.col("UPRN").is_in(uprns_in_hn))
        .then(True)
        .otherwise(False)
        .alias("in_hn"),
    )
    .join(outdoor_space_df, how="left", on="UPRN")
    .join(epc_df, how="left", on="UPRN")
)

# %%
features_df.null_count() / len(features_df)

# %%
features_df.shape

# %% [markdown]
# ## Assess how many EPC records missing
#
# As most of the data we are missing for UPRNs originates from EPC, we want to get a sense of how many EPC records are missing per LSOA.

# %%
# Get Plymouth LA code
plymouth_ladcd = la_boundaries_gdf[la_boundaries_gdf["LAD23NM"] == "Plymouth"][
    "LAD23CD"
].values[0]

print("Loading postcode lookup...")
postcode_lookup_df = output_areas.load_transform_df_area_info(
    use_cols=[
        "pcd",
        "lsoa11",
        "msoa11",
        "lsoa21",
        "msoa21",
        "ru11ind",
        "oslaua",
        "oa21",
        "imd",
    ]
)

print("\nLoading ONS census number of household per LSOA...")
n_households_df = get_datasets.get_df_ons_number_of_households()

# Join output area codes to EPC data on postcode
epc_df = epc_df.join(
    postcode_lookup_df[
        [
            "POSTCODE",
            "oa21",
            "lsoa",
            "msoa",
            "lad_code",
            "country_code",
            "ruc_two_fold",
            "ruc_EW_ten_fold",
            "imd",
        ]
    ],
    how="left",
    on="POSTCODE",
)

# Get LSOAs in Plymouth
plymouth_lsoas = postcode_lookup_df.filter(pl.col("lad_code") == plymouth_ladcd)[
    "lsoa"
].unique()

# Filter EPC to Plymouth LSOAs and get counts per LSOA of households according to EPC data vs census
epc_counts_df = (
    epc_df.filter(pl.col("lsoa").is_in(plymouth_lsoas))
    .group_by("lsoa")
    .agg(pl.col("UPRN").count())
    .join(
        n_households_df.select(["mnemonic", "2021"]),
        how="left",
        left_on="lsoa",
        right_on="mnemonic",
    )
    .rename({"UPRN": "EPC_records_count", "2021": "census_count"})
    .with_columns(
        (pl.col("census_count") - pl.col("EPC_records_count")).alias("count_diff")
    )
)

# Plot differences in household counts
plt.hist(epc_counts_df["count_diff"], bins=20)
plt.title(
    "Distribution of diffs between 2021 census count of households and EPC records\nin Plymouth LSOAs"
)
plt.ylabel("Count of LSOAs")
plt.xlabel("Count of households in 2021 census - count of EPC records")
plt.show()

print("Number of EPC records in Plymouth:")
print(epc_counts_df["EPC_records_count"].sum())

print("Number of households in census in Plymouth:")
print(epc_counts_df["census_count"].sum())

print("Number of residential UPRNs in Plymouth:")

print(len(features_df))

# %% [markdown]
# ## Fill missing data - off gas

# %%
# FIND NEAREST POSTCODE TO EACH UPRN
# Remove space from postcode in code point data
code_point_df["POSTCODE"] = code_point_df["postcode"].str.replace(" ", "")

# Find nearest postcode point for all UPRNs within a 1km radius
nearest_postcode_df = pl.from_pandas(
    plymouth_residential_uprns_gdf.sjoin_nearest(
        code_point_df[["POSTCODE", "geometry"]],
        how="left",
        max_distance=1000,
        distance_col="distance_to_postcode_m",
    ).drop(columns="index_right")[["UPRN", "POSTCODE", "distance_to_postcode_m"]]
)

# ----------------------------------------------------------------------------------- #
# ASSIGN OFF GAS LABEL

# Label all UPRNs with on/off gas where possible
off_gas_df = (
    nearest_postcode_df.with_columns(
        # Label postcodes according to on/off gas
        pl.when(pl.col("POSTCODE").is_in(off_gas_list))
        .then(True)
        .otherwise(False)
        .alias("off_gas")
    )
    .group_by("UPRN")
    .agg(
        # Get number of postcodes per UPRN and modal on/off gas value
        pl.col("POSTCODE").alias("postcodes"),
        pl.col("POSTCODE").len().alias("n_postcodes"),
        pl.col("off_gas").mode().alias("modal_off_gas"),
        pl.col("distance_to_postcode_m").first().alias("distance_to_postcode_m"),
    )
    .with_columns(
        # Select single modal on/off gas value
        pl.when(pl.col("modal_off_gas").list.len() == 1).then(
            pl.col("modal_off_gas").list.get(0)
        )
        # If there are multiple modal values, set on/off gas as Null
        .otherwise(None).alias("off_gas"),
        pl.when(pl.col("n_postcodes") == 1).then(pl.col("postcodes").list.get(0))
        # If there are multiple postcode values, set postcode as Null
        .otherwise(None).alias("POSTCODE"),
    )
    .select(["UPRN", "POSTCODE", "distance_to_postcode_m", "off_gas"])
)

# Join postcode to UPRNs and assign on/off gas label based on postcode
features_df = (
    features_df.with_columns(
        pl.when(pl.col("POSTCODE").is_in(off_gas_list))
        .then(True)
        .when(pl.col("POSTCODE").is_not_null())
        .then(False)
        .alias("off_gas")
    )
    .join(
        off_gas_df.select(["UPRN", "POSTCODE", "off_gas", "distance_to_postcode_m"]),
        how="left",
        on="UPRN",
        suffix="_nearest",
    )
    .with_columns(
        (pl.col("POSTCODE").fill_null(pl.col("POSTCODE_nearest"))).alias(
            "use_postcode"
        ),
        pl.col("off_gas").fill_null(pl.col("off_gas_nearest")).alias("filled_off_gas"),
    )
)

# Fill missing off-gas label
features_df = features_df.with_columns(
    pl.when((pl.col("off_gas").is_null()) & (pl.col("fuel_type") == "gas"))
    .then(False)
    .otherwise(pl.col("off_gas"))
    .alias("off_gas")
)

# ----------------------------------------------------------------------------------- #
# EVALUATE ACCURACY OF RESULTS

# Evaluate postcode match accuracy
evaluate_postcode_match_df = (
    features_df.select(
        ["UPRN", "POSTCODE", "POSTCODE_nearest", "distance_to_postcode_m"]
    )
    .filter(pl.col("POSTCODE").is_not_null())
    .with_columns(
        pl.when(pl.col("POSTCODE") == pl.col("POSTCODE_nearest"))
        .then(True)
        .otherwise(False)
        .alias("postcode_match")
    )
)

print(evaluate_postcode_match_df["postcode_match"].value_counts(normalize=True))
print(features_df.filter(pl.col("POSTCODE").is_not_null()).shape)
print(features_df.filter(pl.col("POSTCODE").is_null()).shape)
print(features_df["UPRN"].n_unique())
print(len(features_df))
print(features_df["filled_off_gas"].value_counts(normalize=True))

# Plot distance distribution between incorrect and correct matches
plt.hist(
    evaluate_postcode_match_df.filter(~pl.col("postcode_match"))[
        "distance_to_postcode_m"
    ],
    bins=50,
    density=True,
    alpha=0.5,
    label="Postcode different from EPC",
)
plt.hist(
    evaluate_postcode_match_df.filter(pl.col("postcode_match"))[
        "distance_to_postcode_m"
    ],
    bins=50,
    density=True,
    alpha=0.5,
    label="Postcode matches EPC",
)
plt.title("Distribution of distance from UPRN to postcode point (m)")
plt.xlabel("Distance between UPRN and postcode point (m)")
plt.legend()
plt.show()

# %% [markdown]
# ## Fill missing data - property type

# %%
# Join residential UPRNs with their corresponding buildings
uprns_buildings_df = pl.from_pandas(
    plymouth_residential_uprns_gdf.sjoin(
        plymouth_residential_buildings_gdf[["ID", "geometry", "building_area_m2"]],
        how="left",
        predicate="within",
    )
    .rename(columns={"ID": "building_id"})
    .drop(columns=["index_right", "geometry"])
).join(features_df.select(["UPRN", "property_type"]), how="left", on="UPRN")

# Identify building IDs which have UPRNs missing property type information
missing_property_type_buildings_list = (
    uprns_buildings_df.filter(pl.col("property_type").is_null())["building_id"]
    .unique()
    .to_list()
)

# Aggregate summary property type information for the selected building IDs
property_type_df = (
    uprns_buildings_df.filter(
        pl.col("building_id").is_in(missing_property_type_buildings_list),
    )
    .group_by(pl.col("building_id"))
    .agg(
        pl.col("UPRN").count().alias("n_UPRNs"),
        pl.col("property_type").alias("property_types"),
        pl.col("property_type").null_count().alias("null_count"),
        pl.col("property_type").n_unique().alias("nunique_property_types"),
        pl.col("property_type").mode().alias("modal_property_type"),
        pl.col("property_type").mode().len().alias("n_modal_property_type"),
    )
    .with_columns(
        ((pl.col("n_UPRNs") - pl.col("null_count")) / pl.col("n_UPRNs")).alias(
            "non_null_proportion"
        )
    )
)

print(
    "UPRNs that are missing a property type and are in a building with multiple property type labels:"
)
print(
    property_type_df.filter(pl.col("nunique_property_types") > 2)["null_count"].sum()
    / len(plymouth_residential_uprns_gdf)
)

print(
    "\nUPRNs that are missing a property type and are in a building with multiple MODAL property type labels:"
)
print(
    property_type_df.filter(pl.col("n_modal_property_type") > 1)["null_count"].sum()
    / len(plymouth_residential_uprns_gdf)
)

print(
    "\nUPRNs that are missing a property type and are in a multi-property building with NO property type labels:"
)
print(
    property_type_df.filter(
        pl.col("n_UPRNs") > 1, pl.col("non_null_proportion") == 0
    ).height
    / len(plymouth_residential_uprns_gdf)
)

# %%
# Get UPRN density of buildings where all UPRNs have property type labels and they all have the same label
density_df = (
    uprns_buildings_df.filter(
        ~pl.col("building_id").is_in(missing_property_type_buildings_list)
    )
    .group_by("building_id")
    .agg(
        pl.col("UPRN").n_unique().alias("n_UPRNs"),
        pl.col("building_area_m2").first().alias("building_area_m2"),
        pl.col("property_type").unique().alias("property_types"),
        pl.col("property_type").n_unique().alias("n_property_types"),
    )
    .filter(pl.col("n_property_types") == 1)
    .with_columns(
        pl.col("property_types").list.get(0).alias("property_type"),
        (pl.col("n_UPRNs") / (pl.col("building_area_m2") / 1000)).alias(
            "UPRN_density_km2"
        ),
    )
)

# Plot UPRN density distribution for each property type
fig = plt.figure(figsize=(10, 5))
for property_type in density_df["property_type"].sort().unique():
    if property_type != "Detached":
        plt.hist(
            density_df.filter(pl.col("property_type") == property_type)[
                "UPRN_density_km2"
            ],
            density=True,
            bins=40,
            alpha=0.5,
            label=f"{property_type},\nN={len(density_df.filter(pl.col('property_type') == property_type))}",
        )
plt.title("Within-building UPRN density in different property types")
plt.xlabel("UPRNs per km2 of building footprint")
plt.legend()

# %%
# Plot building footprint area distribution for each property type
fig = plt.figure(figsize=(10, 5))
for property_type in density_df["property_type"].sort().unique():
    plt.hist(
        density_df.filter(pl.col("property_type") == property_type)["building_area_m2"],
        density=True,
        bins=40,
        alpha=0.5,
        label=f"{property_type},\nN={len(density_df.filter(pl.col('property_type') == property_type))}",
    )
plt.title("Building footprint area in different property types")
plt.xlabel("Building footprint area (m2)")
plt.xlim(0, 500)
plt.legend()

# %%
# CALCULATE NEAREST NEIGHBOUR DISTANCE

# Calculate distance from nearest neighbour for UPRNs with property type label
# Get coordinates of each UPRN
nn_dist_gdf = lat_lon.generate_gdf_uprn_coords(
    uprns_buildings_df.filter(pl.col("property_type").is_not_null())
)

# Convert coordinates to array
coords = np.array(nn_dist_gdf.geometry.map(lambda p: [p.x, p.y]).tolist())

# Find two nearest neighbours to each UPRN
knn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(coords)
knn_dist, knn_idx = knn.kneighbors(coords)

# Get distance of second nearest neighbour (the nearest neighbour is itself)
# The second neighbour will only be itself if there is another neighbour that is 0m away
nn_dist_gdf[list(map("neighbour_{}".format, range(1, 2)))] = (
    nn_dist_gdf.geometry.values.to_numpy()[knn_idx[:, 1:]]
)
nn_dist_gdf["nn_distance_m"] = nn_dist_gdf["geometry"].distance(
    gpd.GeoSeries(nn_dist_gdf["neighbour_1"], crs="EPSG:27700"), align=True
)

# ----------------------------------------------------------------------------------- #
# PLOT RESULTS

# Plot distance of second nearest neighbour per property type
fig = plt.figure(figsize=(10, 5))
for property_type in [
    "Semi-detached",
    "Terraced (including end-terrace)",
    "Flat, maisonette or apartment",
]:
    plt.hist(
        nn_dist_gdf[
            (nn_dist_gdf["property_type"] == property_type)
            & (nn_dist_gdf["nn_distance_m"] <= 20)
        ]["nn_distance_m"],
        density=True,
        bins=40,
        alpha=0.5,
        label=f"{property_type},\nN={len(nn_dist_gdf[nn_dist_gdf['property_type'] == property_type])}",
    )
plt.title("Distance from nearest neighbour in different property types")
plt.xlabel("Distance from nearest neighbour (m)")
plt.xlim(0, 20)
plt.legend()

# %%
# Table of counts & proportions of buildings with different counts of UPRNs per property type
table_df = (
    density_df.group_by(["property_type", "n_UPRNs"])
    .agg(pl.col("building_id").count().alias("n_buildings"))
    .pivot(on="property_type", index="n_UPRNs", values="n_buildings")
    .to_pandas()
)

table_df = table_df.set_index("n_UPRNs").sort_index()[
    [
        "Detached",
        "Semi-detached",
        "Terraced (including end-terrace)",
        "Flat, maisonette or apartment",
        "Caravan or other mobile or temporary structure",
    ]
]
table_df / table_df.sum() * 100

# %%
table_df.div(table_df.sum(axis=1), axis=0) * 100

# %%
# ----------------------------------------------------------------------------------- #
# CALCULATE NEAREST NEIGHBOUR DISTANCE

# Calculate distance from nearest neighbour for all UPRNs
# Get coordinates of each UPRN
nn_dist_gdf = lat_lon.generate_gdf_uprn_coords(uprns_buildings_df)

# Convert coordinates to array
coords = np.array(nn_dist_gdf.geometry.map(lambda p: [p.x, p.y]).tolist())

# Find two nearest neighbours to each UPRN
knn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(coords)
knn_dist, knn_idx = knn.kneighbors(coords)

# Get distance of second nearest neighbour (the nearest neighbour is itself)
# The second neighbour will only be itself if there is another neighbour that is 0m away
nn_dist_gdf[list(map("neighbour_{}".format, range(1, 2)))] = (
    nn_dist_gdf.geometry.values.to_numpy()[knn_idx[:, 1:]]
)
nn_dist_gdf["nn_distance_m"] = nn_dist_gdf["geometry"].distance(
    gpd.GeoSeries(nn_dist_gdf["neighbour_1"], crs="EPSG:27700"), align=True
)

# ----------------------------------------------------------------------------------- #
# ASSIGN PROPERTY TYPE LABELS

# Apply rules to determine property type
nn_dist_df = pl.from_pandas(nn_dist_gdf.drop(columns=["geometry", "neighbour_1"]))
density_df = uprns_buildings_df.group_by("building_id").agg(
    pl.col("UPRN").n_unique().alias("n_UPRNs")
)

nn_dist_df = (
    nn_dist_df.join(density_df, how="left", on="building_id")
    .with_columns(
        pl.when(pl.col("nn_distance_m") <= 2.5)
        .then(pl.lit("Flat, maisonette or apartment"))
        .when(pl.col("n_UPRNs") == 1)
        .then(pl.lit("Detached"))
        .when(pl.col("n_UPRNs") == 2)
        .then(pl.lit("Semi-detached"))
        .otherwise(pl.lit("Terraced (including end-terrace)"))
        .alias("predicted_property_type")
    )
    .with_columns(
        (pl.col("property_type") == pl.col("predicted_property_type")).alias(
            "correct_prediction"
        )
    )
)

property_types = [
    "Detached",
    "Semi-detached",
    "Terraced (including end-terrace)",
    "Flat, maisonette or apartment",
]

# ----------------------------------------------------------------------------------- #
# CHECK RESULTS OF PREDICTIONS

# Proportions of correct predictions per property type
print(
    nn_dist_df.filter(
        pl.col("correct_prediction").is_not_null(),
        pl.col("property_type") != "Caravan or other mobile or temporary structure",
    )["correct_prediction"].value_counts(normalize=True)
)

# Proportions of correct predictions per property type and UPRN count per building
results_df = (
    nn_dist_df.filter(
        pl.col("correct_prediction").is_not_null(),
        pl.col("property_type") != "Caravan or other mobile or temporary structure",
    )
    .group_by(["property_type", "correct_prediction"])
    .agg(pl.col("UPRN").count())
    .pivot(on="property_type", index="correct_prediction", values="UPRN")
    .with_columns((pl.col(property_types) / pl.col(property_types).sum()) * 100)
    .select(["correct_prediction"] + property_types)
)

print(results_df)

# Proportions of predicted property type for incorrectly predicted detached properties
print(
    nn_dist_df.filter(
        pl.col("correct_prediction").is_not_null(),
        pl.col("property_type") == "Detached",
        ~pl.col("correct_prediction"),
    )["predicted_property_type"].value_counts(normalize=True)
)

# Proportions of predicted property type for incorrectly predicted semi-detached properties
print(
    nn_dist_df.filter(
        pl.col("correct_prediction").is_not_null(),
        pl.col("property_type") == "Semi-detached",
        ~pl.col("correct_prediction"),
    )["predicted_property_type"].value_counts(normalize=True)
)

# Join predicted property type to features df
features_df = features_df.join(
    nn_dist_df.select(["UPRN", "predicted_property_type"]), how="left", on="UPRN"
)

# %% [markdown]
# ## Fill missing data - tenure type

# %%
# Load census data
print("Loading age band data...")
age_bands_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_age_bands_OA_plymouth.csv",
    skip_rows=4,
)

print("\nLoading disability data...")
disability_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_disability_OA_plymouth.csv",
    skip_rows=6,
)

print("\nLoading economic status data...")
economic_status_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_economic_status_OA_plymouth.csv",
    skip_rows=6,
)

print("\nLoading hours worked data...")
hours_worked_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_hours_worked_OA_plymouth.csv",
    skip_rows=6,
)

print("\nLoading student binary data...")
student_binary_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_student_binary_OA_plymouth.csv",
    skip_rows=6,
)

print("\nLoading tenure data...")
oa_tenure_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_tenure_OA_plymouth.csv",
    skip_rows=6,
)

# %%
# Preprocess census data to generate features for predictive model
# Process age band data
age_bands_df = (
    age_bands_df.filter(
        pl.col("2021 output area").is_not_null(), pl.col("2021 output area") != ""
    )
    .rename(
        {
            k: v
            for k, v in zip(
                age_bands_df.columns,
                [col.lower().replace(" ", "_") for col in age_bands_df.columns],
            )
        }
    )
    .rename({"2021_output_area": "oa_code"})
    .drop("total")
    .with_columns(
        pl.sum_horizontal(
            "aged_16_to_19_years", "aged_20_to_24_years", "aged_25_to_34_years"
        ).alias("prop_aged_16_to_35"),
        pl.sum_horizontal("aged_35_to_49_years", "aged_50_to_64_years").alias(
            "prop_aged_35_to_64"
        ),
        pl.sum_horizontal(
            "aged_65_to_74_years", "aged_75_to_84_years", "aged_85_years_and_over"
        ).alias("prop_aged_65_and_over"),
    )
    .with_columns(
        pl.sum_horizontal(
            "prop_aged_16_to_35", "prop_aged_35_to_64", "prop_aged_65_and_over"
        ).alias("total")
    )
    .select(
        [
            "oa_code",
            "prop_aged_16_to_35",
            "prop_aged_35_to_64",
            "prop_aged_65_and_over",
            "total",
        ]
    )
    .with_columns((pl.col(pl.Int64) / pl.col("total")))
    .drop("total")
)

# Process disability data
disability_df = (
    disability_df.filter(
        pl.col("2021 output area").is_not_null(), pl.col("2021 output area") != ""
    )
    .rename({"2021 output area": "oa_code"})
    .with_columns(
        (
            pl.col("Disabled under the Equality Act")
            / pl.col("Total: All usual residents")
        ).alias("prop_disabled_residents")
    )
    .select(["oa_code", "prop_disabled_residents"])
)

# Process economic status data
economic_status_df = (
    economic_status_df.filter(
        pl.col("2021 output area").is_not_null(), pl.col("2021 output area") != ""
    )
    .rename({"2021 output area": "oa_code"})
    .with_columns(
        (
            pl.col("Economically inactive")
            / pl.col("Total: All usual residents aged 16 years and over")
        ).alias("prop_economically_inactive_adults")
    )
    .select(["oa_code", "prop_economically_inactive_adults"])
)

# Process hours worked data
hours_worked_df = (
    hours_worked_df.filter(
        pl.col("2021 output area").is_not_null(), pl.col("2021 output area") != ""
    )
    .rename({"2021 output area": "oa_code"})
    .with_columns(
        (
            pl.col("Full-time")
            / pl.col(
                "Total: All usual residents aged 16 years and over in employment the week before the census"
            )
        ).alias("prop_employed_adults_working_full_time")
    )
    .select(["oa_code", "prop_employed_adults_working_full_time"])
)

# Process student binary
student_binary_df = (
    student_binary_df.filter(
        pl.col("2021 output area").is_not_null(), pl.col("2021 output area") != ""
    )
    .rename({"2021 output area": "oa_code"})
    .with_columns((pl.col("Student") / pl.col("Total")).alias("prop_students"))
    .select(["oa_code", "prop_students"])
)

# Process tenure
oa_tenure_df = (
    oa_tenure_df.filter(
        pl.col("2021 output area").is_not_null(), pl.col("2021 output area") != ""
    )
    .rename({"2021 output area": "oa_code"})
    .select(
        [
            "oa_code",
            "Total: All households",
            "Owned",
            "Shared ownership",
            "Social rented",
            "Private rented",
            "Lives rent free",
        ]
    )
    .with_columns(
        (
            pl.sum_horizontal("Owned", "Shared ownership")
            / pl.col("Total: All households")
        ).alias("prop_owner_occupied"),
        (pl.col("Social rented") / pl.col("Total: All households")).alias(
            "prop_social_rented"
        ),
    )
    .select(["oa_code", "prop_owner_occupied", "prop_social_rented"])
)

# Join census datasets together
census_df = oa_tenure_df
for df in [
    age_bands_df,
    disability_df,
    economic_status_df,
    hours_worked_df,
    student_binary_df,
]:
    census_df = census_df.join(df, how="left", on="oa_code")

# %%
# CREATE DATAFRAME WITH FEATURES TO TRAIN AND TEST MODEL
model_df = (
    epc_df.select(
        [
            "UPRN",
            "property_type",
            "TENURE",
            "oa21",
        ]
    )
    .join(census_df, how="left", left_on="oa21", right_on="oa_code")
    .join(
        features_df.select(["UPRN", "in_cons_area", "in_listed_building"]),
        how="left",
        on="UPRN",
    )
    .filter(
        pl.all_horizontal(pl.all().is_not_null()),
        pl.col("property_type") != "Caravan or other mobile or temporary structure",
        pl.col("TENURE") != "unknown",
    )
    .drop(["oa21", "prop_aged_35_to_64"])
    .to_pandas()
)

X = model_df.drop(columns=["TENURE"])
y = model_df[["TENURE"]]

# One-hot encode property type feature
enc = OneHotEncoder()
onehotenc_df = pd.DataFrame(enc.fit_transform(X[["property_type"]]).toarray())
onehotenc_df.columns = list(col.lower().replace(" ", "_") for col in enc.categories_[0])

# Join encodings to X
X = X.join(onehotenc_df, how="left").drop(columns=["property_type", "detached", "UPRN"])

# ----------------------------------------------------------------------------------- #
# TRAIN AND TEST MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# Fit and predict with K-nearest neighbours classifier model
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train)
print("K Nearest Neighbour classifier accuracy score:")
print(classifier.score(X_test, y_test))
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)

# Create dataframe with results of predictions and probabilities
results_df = (
    pl.DataFrame(
        {
            "actual": y_test["TENURE"],
            "predicted": y_pred,
            "0": y_pred_prob[:, 0],
            "1": y_pred_prob[:, 1],
            "2": y_pred_prob[:, 2],
        }
    )
    .rename({k: v for k, v in zip(["0", "1", "2"], classifier.classes_)})
    .with_columns(
        (pl.col("actual") == pl.col("predicted")).alias("correct_prediction"),
        max_prob=pl.concat_list(
            "owner-occupied", "rental (private)", "rental (social)"
        ).list.max(),
    )
)

# classifier = RandomForestClassifier()
# classifier.fit(X_train, y_train)
# print("\nRandom Forest classifier accuracy score:")
# print(classifier.score(X_test, y_test))

# ----------------------------------------------------------------------------------- #
# PLOT RESULTS

# Plot probability of predictions based on prediction outcome
plot_df = results_df.group_by(["correct_prediction", "max_prob"]).agg(
    pl.col("predicted").count().alias("n_predictions")
)
fig = plt.figure(figsize=(10, 5))
plot_df = (
    plot_df.pivot(
        on="correct_prediction",
        index="max_prob",
        values="n_predictions",
    )
    .rename({"true": "Correct prediction", "false": "Incorrect prediction"})
    .sort(by="max_prob", descending=False)
    .to_pandas()
)

plot_df.index = plot_df["max_prob"]
plot_df = plot_df.drop("max_prob", axis=1)

plot_df.plot(kind="bar")

plt.ylabel("Count of predictions")
plt.xlabel("Prediction probability")
plt.title("Prediction probability by prediction accuracy")
plt.show()

# %%
# USE MODEL TO PREDICT TENURE TYPE
# Load output area geospatial boundaries
oa_boundaries_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/Output_Areas_2021_EW_BFE_V9_6122634609897870819/OA_2021_EW_BFE_V9.shp"
)

# Assign UPRNs to their output areas
uprns_with_oa_df = plymouth_residential_uprns_gdf.sjoin(
    oa_boundaries_gdf[["OA21CD", "geometry"]], how="left", predicate="intersects"
).drop(columns="index_right")
features_df = features_df.join(
    pl.from_pandas(uprns_with_oa_df[["UPRN", "OA21CD"]]), how="left", on="UPRN"
)
pd_features_df = features_df.to_pandas()

# One hot encode property type data
onehotenc_df = pd.DataFrame(
    enc.transform(
        pd_features_df[["predicted_property_type"]].rename(
            columns={"predicted_property_type": "property_type"}
        )
    ).toarray()
)
onehotenc_df.columns = list(col.lower().replace(" ", "_") for col in enc.categories_[0])
X = (
    pd_features_df.join(onehotenc_df, how="left")
    .drop(columns=["property_type", "detached"])
    .merge(census_df.to_pandas(), how="left", left_on="OA21CD", right_on="oa_code")[
        X.columns
    ]
)

# Run model and get results
y_pred = classifier.predict(X)
y_pred_prob = classifier.predict_proba(X)

results_df = pd.DataFrame(
    {
        "predicted_tenure": y_pred,
        "0": y_pred_prob[:, 0],
        "1": y_pred_prob[:, 1],
        "2": y_pred_prob[:, 2],
    }
).rename(columns={k: v for k, v in zip(["0", "1", "2"], classifier.classes_)})

# Join predicted tenure type to features dataset
features_df = pl.from_pandas(pd_features_df.join(results_df, how="left"))
features_df = features_df.with_columns(
    tenure_prob=pl.concat_list(
        "owner-occupied", "rental (private)", "rental (social)"
    ).list.max(),
).drop(["owner-occupied", "rental (private)", "rental (social)"])
