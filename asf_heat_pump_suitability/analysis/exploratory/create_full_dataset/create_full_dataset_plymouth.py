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

from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from asf_heat_pump_suitability.utils import geo_utils, save_utils
from asf_heat_pump_suitability.getters import get_datasets
from asf_heat_pump_suitability.pipeline.prepare_features import (
    lat_lon,
    off_gas,
    output_areas,
)
from asf_heat_pump_suitability.pipeline.transform import outdoor_space
from asf_heat_pump_suitability.analysis.flats_on_fossils.features import fuel_type
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_sample
import folium
import config

# %% [markdown]
# ## Load data
# Some datasets are for Great Britain / England and some are for Plymouth only to increase download speed.

# %%
# View OpenMap Local dataset layers for GB
fiona.listlayers(config["data"]["geodata"]["gb_os_openmap_local"])

# %%
print("LOADING DATASETS TO GET RESIDENTIAL UPRNS FOR PLYMOUTH...")
print("Loading OS UPRN...")
# Source: https://www.ordnancesurvey.co.uk/products/os-open-uprn
os_uprn_df = get_datasets.get_df_osopen_uprn_latlon()

print("\nLoading LA boundaries...")
# Source: https://www.data.gov.uk/dataset/288458f7-7789-47d0-80d4-ffdf746c6b75/local-authority-districts-december-2023-boundaries-uk-bfe
la_boundaries_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/source_data/Local_Authority_Districts_December_2023_Boundaries_UK_BFE_-2600600853110041429/LAD_DEC_2023_UK_BFE.shp"
)

print("\nLoading OS OpenMap Local - Buildings...")
# Source: https://www.ordnancesurvey.co.uk/products/os-open-map-local
os_openmap_building_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/OS OpenMap Local (ESRI Shape File) SX/data/SX_Building.shp"
)

print("\nLoading OS OpenMap Local - Important Buildings...")
# Source: https://www.ordnancesurvey.co.uk/products/os-open-map-local
os_openmap_importantbuilding_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/OS OpenMap Local (ESRI Shape File) SX/data/SX_ImportantBuilding.shp"
)

print("\nLoading OS OpenMap Local - Railways...")
# Source: https://www.ordnancesurvey.co.uk/products/os-open-map-local
os_openmap_railway_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/OS OpenMap Local (ESRI Shape File) SX/data/SX_RailwayStation.shp"
)

# --------------------------------------------------------------------------#

print("\nLOADING PROPERTY-LEVEL FEATURE DATASETS...")
print("\nLoading listed buildings...")
# Source: https://opendata-historicengland.hub.arcgis.com/datasets/historicengland::national-heritage-list-for-england-nhle/explore?layer=0&location=50.370395%2C-4.182394%2C16.14
listed_buildings_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/National_Heritage_List_for_England_NHLE_v02_VIEW_-464524051049198649/Listed_Building_points.shp"
)

print("\nLoading building conservation areas...")
# Source: https://www.planning.data.gov.uk/dataset/conservation-area#
cons_areas_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/conservation-area (1).geojson"
)

print("\nLoading existing / planned HN zones...")
# Source: HNZ analysis pipeline
hn_zones_gdf = gpd.read_file(
    config["data"]["geodata"]["heat_network_zones"]["plymouth"],
    layer="heat-network-zone-map-Plymouth",
)

print("\nLoading off gas postcodes...")
# Source: https://www.xoserve.com/help-centre/supply-points-metering/supply-point-administration-spa/
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
print("\nLoading OS Code-Point Open (all postcode units in GB)...")
# Source: https://www.ordnancesurvey.co.uk/products/code-point-open
code_point_df = gpd.read_file(
    config["data"]["geodata"]["gb_code_points"],
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
        listed_buildings_gdf, how="inner", max_distance=10  # 10 metres
    )["UPRN"]
    .unique()
    .tolist()
)

# UPRNs in building conservation areas
cons_area_uprns = plymouth_residential_uprns_gdf.sjoin(
    cons_areas_gdf.to_crs(epsg=27700), how="inner", predicate="intersects"
)["UPRN"].tolist()

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
    pl.col("UPRN").str.replace(".0", "", literal=True)
    # This will cast filled UPRNs made up of Building Ref Number and Address back to null
    .cast(pl.Int64, strict=False).alias("UPRN"),
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
        .alias("in_conservation_area"),
        pl.when(pl.col("UPRN").is_in(uprns_in_hn))
        .then(True)
        .otherwise(False)
        .alias("in_heat_network_zone"),
    )
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
        # "LSOA21NM"
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

# Find nearest postcode point within a 1km radius for all UPRNs
nearest_postcode_df = pl.from_pandas(
    plymouth_residential_uprns_gdf.sjoin_nearest(
        code_point_df[["POSTCODE", "geometry"]],
        how="left",
        max_distance=1000,
        distance_col="distance_to_postcode_m",  # distance in metres
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
        # All nearest postcodes will be the same distance so we just select the first to keep one
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

# Identify building IDs which have one or more UPRNs missing property type information
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
    "Proportion of UPRNs that are missing a property type and are in a building with multiple property type labels:"
)
print(
    property_type_df.filter(pl.col("nunique_property_types") > 2)["null_count"].sum()
    / len(plymouth_residential_uprns_gdf)
)

print(
    "\nProportion of UPRNs that are missing a property type and are in a building with multiple MODAL property type labels:"
)
print(
    property_type_df.filter(pl.col("n_modal_property_type") > 1)["null_count"].sum()
    / len(plymouth_residential_uprns_gdf)
)

print(
    "\nProportion of UPRNs that are missing a property type and are in a multi-property building with NO property type labels:"
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
# Calculate NN distance for properties where property type is known and unique within a building

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
# Percentage of properties with n_UPRNs, per property type
table_df / table_df.sum() * 100

# %%
# Percentage of all properties which are X property type per n_UPRNs
table_df.div(table_df.sum(axis=1), axis=0) * 100

# %%
# ----------------------------------------------------------------------------------- #
# APPLY MODEL - CALCULATE NEAREST NEIGHBOUR DISTANCE FOR ALL UPRNs

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
# LOAD CENSUS DATA
print("Loading age band data...")
# Source: https://www.nomisweb.co.uk/datasets/c2021pp012
age_bands_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_age_bands_OA_plymouth.csv",
    skip_rows=4,
)

print("\nLoading disability data...")
# Source: https://www.nomisweb.co.uk/datasets/c2021ts038
disability_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_disability_OA_plymouth.csv",
    skip_rows=6,
)

print("\nLoading economic status data...")
# Source: https://www.nomisweb.co.uk/datasets/c2021ts066
economic_status_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_economic_status_OA_plymouth.csv",
    skip_rows=6,
)

print("\nLoading hours worked data...")
# Source: https://www.nomisweb.co.uk/datasets/c2021ts059
hours_worked_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_hours_worked_OA_plymouth.csv",
    skip_rows=6,
)

print("\nLoading student binary data...")
# Source: https://www.nomisweb.co.uk/datasets/c2021ts068
student_binary_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_student_binary_OA_plymouth.csv",
    skip_rows=6,
)

print("\nLoading tenure data...")
# Source: https://www.nomisweb.co.uk/datasets/c2021ts054
oa_tenure_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/2021Census_tenure_OA_plymouth.csv",
    skip_rows=6,
)

# %%
# PREPROCESS CENSUS DATA TO GENERATE FEATURES FOR PREDICTIVE MODEL
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
# census_df = oa_tenure_df
census_df = age_bands_df
for df in [
    # age_bands_df,
    disability_df,
    economic_status_df,
    hours_worked_df,
    student_binary_df,
]:
    census_df = census_df.join(df, how="left", on="oa_code")

# %%
# Load output area geospatial boundaries
# Source: https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-boundaries-ew-bfe-v9/about
oa_boundaries_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/Output_Areas_2021_EW_BFE_V9_6122634609897870819/OA_2021_EW_BFE_V9.shp"
)

# Assign UPRNs to their output areas
uprns_with_oa_df = plymouth_residential_uprns_gdf.sjoin(
    oa_boundaries_gdf[["OA21CD", "LSOA21CD", "geometry"]],
    how="left",
    predicate="intersects",
).drop(columns="index_right")

# Join output areas to features
features_df = features_df.join(
    pl.from_pandas(uprns_with_oa_df[["UPRN", "OA21CD", "LSOA21CD"]]),
    how="left",
    on="UPRN",
).with_columns(pl.col("TENURE").replace("unknown", None))

# %%
# CALCULATE PROPORTIONS OF TENURE TYPES FOR NEIGHBOURS WITHIN 100m RADIUS FOR EACH UPRN
nn_tenure_df = features_df.select(
    [
        "UPRN",
        "TENURE",
        "OA21CD",
    ]
).join(
    os_uprn_df.select(["UPRN", "X_COORDINATE", "Y_COORDINATE"]), how="left", on="UPRN"
)

# Calculate distance from nearest neighbour for all UPRNs
# Get coordinates of each UPRN
nn_tenure_df = lat_lon.generate_gdf_uprn_coords(
    nn_tenure_df, usecols=["UPRN", "TENURE", "X_COORDINATE", "Y_COORDINATE"]
)

# Convert coordinates to array
coords = np.array(nn_tenure_df.geometry.map(lambda p: [p.x, p.y]).tolist())

# Find all neighbours within 100m radius of each UPRN
knn = NearestNeighbors(radius=100, algorithm="kd_tree").fit(coords)
knn_dist, knn_idx = knn.radius_neighbors(coords)

# Get the tenures of the neighbours within 100m
nn_tenure_df["nearest_tenures"] = [
    nn_tenure_df.TENURE.values[knn_idx[i]] for i in range(0, len(knn_idx))
]

# Get the UPRNs of the neighbours within 100m
nn_tenure_df["nearest_UPRNs"] = [
    nn_tenure_df.UPRN.values[knn_idx[i]] for i in range(0, len(knn_idx))
]

nn_tenure_df = pd.DataFrame(nn_tenure_df).explode(
    column=["nearest_tenures", "nearest_UPRNs"]
)
nn_tenure_df = (
    nn_tenure_df[
        # Filter out neighbours which are 'self' and filter out any neighbours with no tenure
        (nn_tenure_df["UPRN"] != nn_tenure_df["nearest_UPRNs"])
        & (nn_tenure_df["nearest_tenures"].notnull())
        # Get the counts of neighbours with each tenure type per UPRN
    ]
    .groupby(["UPRN", "nearest_tenures"])
    .agg(
        count=pd.NamedAgg(column="UPRN", aggfunc="count"),
    )
    .reset_index()
)

# Calculate the proportion of neighbours with each tenure type
total_tenure_df = nn_tenure_df.groupby("UPRN").agg(
    total=pd.NamedAgg(column="count", aggfunc="sum")
)
nn_tenure_df = nn_tenure_df.merge(total_tenure_df, how="left", on="UPRN")
nn_tenure_df["prop_neighbours"] = nn_tenure_df["count"] / nn_tenure_df["total"]

nn_tenure_df = pl.from_pandas(
    nn_tenure_df.pivot(
        index="UPRN", columns="nearest_tenures", values="prop_neighbours"
    )
    .fillna(0)
    .reset_index()
).rename(
    {
        "owner-occupied": "prop_nn_owner_occupied",
        "rental (private)": "prop_nn_private_rental",
        "rental (social)": "prop_nn_social_rental",
    }
)

# ----------------------------------------------------------------------------------- #
# CREATE DATAFRAME WITH FEATURES TO TRAIN AND TEST MODEL
model_df = (
    features_df.select(
        [
            "UPRN",
            "property_type",
            "TENURE",
            "OA21CD",
        ]
    )
    .join(census_df, how="left", left_on="OA21CD", right_on="oa_code")
    .join(
        os_uprn_df.select(["UPRN", "X_COORDINATE", "Y_COORDINATE"]),
        how="left",
        on="UPRN",
    )
    .join(nn_tenure_df.drop("prop_nn_private_rental"), how="left", on="UPRN")
    .join(
        features_df.select(["UPRN", "in_conservation_area", "in_listed_building"]),
        how="left",
        on="UPRN",
    )
    .filter(
        pl.all_horizontal(pl.all().is_not_null()),
        pl.col("property_type") != "Caravan or other mobile or temporary structure",
        pl.col("TENURE") != "unknown",
    )
    .drop(["OA21CD", "prop_aged_35_to_64"])
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

print(y_train["TENURE"].value_counts(normalize=True))
print(y_test["TENURE"].value_counts(normalize=True))

# Fit and predict with K-nearest neighbours classifier model
# classifier = KNeighborsClassifier(n_neighbors=10)
# classifier.fit(X_train, y_train)
# print("K Nearest Neighbour classifier accuracy score:")
# print(classifier.score(X_test, y_test))
# y_pred = classifier.predict(X_test)
# y_pred_prob = classifier.predict_proba(X_test)

# Fit and predict with Random Forest classifier model
classifier = RandomForestClassifier(
    class_weight="balanced"
)  # balance class weights due to imbalanced target
classifier.fit(X_train, y_train)
print("\nRandom Forest classifier accuracy score:")
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
    # Rename columns with class names
    .rename({k: v for k, v in zip(["0", "1", "2"], classifier.classes_)}).with_columns(
        (pl.col("actual") == pl.col("predicted")).alias("correct_prediction"),
        # Get maximum class prediction probability across all predictions for a given UPRN
        max_prob=pl.concat_list(
            "owner-occupied", "rental (private)", "rental (social)"
        ).list.max(),
    )
)

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

bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
labels = [
    "(0.3-0.4]",
    "(0.4-0.5]",
    "(0.5-0.6]",
    "(0.6-0.7]",
    "(0.7-0.8]",
    "(0.8-0.9]",
    "(0.9-1]",
]
plot_df["max_prob_group"] = pd.cut(
    plot_df["max_prob"], bins=bins, labels=labels, right=True
)

plot_df = plot_df.groupby("max_prob_group").sum()
plot_df = plot_df.drop("max_prob", axis=1)

plot_df.plot(kind="bar")

plt.ylabel("Count of predictions")
plt.xlabel("Prediction probability")
plt.title("Prediction probability by prediction accuracy")
plt.show()

# %%
# Plot stacked bar chart with proportions of prediction probability vs prediction accuracy
plot_df = plot_df.div(plot_df.sum(axis=1), axis=0) * 100  # convert to 0‑100 %

ax = plot_df.plot(
    kind="bar", stacked=True, figsize=(10, 6), width=0.8, edgecolor="none"
)

ax.set_ylabel("Percentage of predictions")
ax.set_xlabel("Probability of predictions")
ax.set_title("Prediction probability by prediction accuracy")

ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="y", linestyle=":", linewidth=0.5)
plt.tight_layout()
plt.show()

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

tenure_mapping = {"owner-occupied": 0, "rental (private)": 1, "rental (social)": 2}
y_true = np.array(y_test["TENURE"].map(tenure_mapping))

y_pred_arr = np.copy(y_pred)
for k, v in tenure_mapping.items():
    y_pred_arr = [tenure_mapping.get(k) for k in y_pred]

matrix = confusion_matrix(list(y_true), list(y_pred_arr), normalize="true")
print(matrix.diagonal() / matrix.sum(axis=1))

sns.heatmap(
    matrix,
    annot=True,
    cmap="YlGnBu",
    xticklabels=tenure_mapping.keys(),
    yticklabels=tenure_mapping.keys(),
)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix", fontsize=16)
plt.show()

# %%
# FEATURE ENGINEERING
# PLOT CORRELATION HEATMAP
import seaborn as sns

corr = X.corr()
plt.figure(figsize=(15, 8))
ax = sns.heatmap(corr, annot=True)

# %% [markdown]
# ### ------- Test feature engineering for tenure prediction model --------
#
# All the code here is copied directly from an sklearn tutorial with some minor adjustments: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
#
# Ultimately, I decided not to proceed with feature engineering as the original feature importance looks consistent and good through permutation and engineering didn't improve the accuracy or confusion matrix for the model.

# %%
# RANDOM FOREST CLASSIFIER FEATURE IMPORTANCE WITH NO FEATURE ENGINEERING
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils.fixes import parse_version


def plot_permutation_importance(clf, X, y, ax):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    # `labels` argument in boxplot is deprecated in matplotlib 3.9 and has been
    # renamed to `tick_labels`. The following code handles this, but as a
    # scikit-learn user you probably can write simpler code by using `labels=...`
    # (matplotlib < 3.9) or `tick_labels=...` (matplotlib >= 3.9).
    tick_labels_parameter_name = (
        "tick_labels"
        if parse_version(matplotlib.__version__) >= parse_version("3.9")
        else "labels"
    )
    tick_labels_dict = {tick_labels_parameter_name: X.columns[perm_sorted_idx]}
    ax.boxplot(result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
    ax.axvline(x=0, color="k", linestyle="--")
    return ax


mdi_importances = pd.Series(classifier.feature_importances_, index=X_train.columns)
tree_importance_sorted_idx = np.argsort(classifier.feature_importances_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
mdi_importances.sort_values().plot.barh(ax=ax1)
ax1.set_xlabel("Gini importance")
plot_permutation_importance(classifier, X_train, y_train, ax2)
ax2.set_xlabel("Decrease in accuracy score")
fig.suptitle(
    "Impurity-based vs. permutation importances on multicollinear features (train set)"
)
_ = fig.tight_layout()

# %%
# PLOT CORRELATED FEATURES IN HEATMAP AND DENDROGRAM
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
_ = fig.tight_layout()

# %%
# PICK A THRESHOLD FROM DENDROGRAM AND PERFORM HEIRARCHICAL CLUSTERING - DETERMINE A FEATURE FROM EACH CLUSTER TO KEEP
from collections import defaultdict

# Threshold chosen from manual inspection of dendrogram
threshold = 1

cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_features_names = X.columns[selected_features]

X_train_sel = X_train[selected_features_names]
X_test_sel = X_test[selected_features_names]

clf_selected_features = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
clf_selected_features.fit(X_train_sel, y_train)
print(
    "Baseline accuracy on test data with features removed:"
    f" {clf_selected_features.score(X_test_sel, y_test):.3}"
)

fig, ax = plt.subplots(figsize=(7, 6))
plot_permutation_importance(clf_selected_features, X_test_sel, y_test, ax)
ax.set_title("Permutation Importances on selected subset of features\n(test set)")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

tenure_mapping = {"owner-occupied": 0, "rental (private)": 1, "rental (social)": 2}
y_true = np.array(y_test["TENURE"].map(tenure_mapping))
y_pred = clf_selected_features.predict(X_test_sel)

y_pred_arr = np.copy(y_pred)
for k, v in tenure_mapping.items():
    y_pred_arr[y_pred == k] = v

matrix = confusion_matrix(list(y_true), list(y_pred_arr), normalize="true")
print(matrix.diagonal() / matrix.sum(axis=1))

sns.heatmap(
    matrix,
    annot=True,
    cmap="YlGnBu",
    xticklabels=tenure_mapping.keys(),
    yticklabels=tenure_mapping.keys(),
)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix", fontsize=16)
plt.show()

# %% [markdown]
# ### -------- End of feature engineering test --------

# %%
# USE MODEL TO PREDICT TENURE TYPE
features = [
    "prop_aged_16_to_35",
    "prop_aged_65_and_over",
    "prop_disabled_residents",
    "prop_economically_inactive_adults",
    "prop_employed_adults_working_full_time",
    "prop_students",
    "X_COORDINATE",
    "Y_COORDINATE",
    "prop_nn_owner_occupied",
    "prop_nn_social_rental",
    "in_conservation_area",
    "in_listed_building",
]

# Join output areas to features
pd_features_df = (
    # features_df.join(
    # pl.from_pandas(uprns_with_oa_df[["UPRN", "OA21CD", "LSOA21CD"]]), how="left", on="UPRN"
    # )
    features_df.join(
        os_uprn_df.select(["UPRN", "X_COORDINATE", "Y_COORDINATE"]),
        how="left",
        on="UPRN",
    )
    .join(nn_tenure_df, how="left", on="UPRN")
    .join(census_df, how="left", left_on="OA21CD", right_on="oa_code")
    .with_columns(
        pl.col("property_type")
        .fill_null(pl.col("predicted_property_type"))
        .alias("use_property_type")
    )
    .filter(
        pl.col("use_property_type") != "Caravan or other mobile or temporary structure",
        pl.all_horizontal(pl.col(features).is_not_null()),
    )
    .to_pandas()
)

# One hot encode property type data
onehotenc_df = pd.DataFrame(
    enc.transform(
        pd_features_df[["use_property_type"]].rename(
            columns={"use_property_type": "property_type"}
        )
    ).toarray()
)
onehotenc_df.columns = list(col.lower().replace(" ", "_") for col in enc.categories_[0])

X = pd_features_df.join(onehotenc_df, how="left")[X.columns]

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

# # Join predicted tenure type to features dataset
predicted_tenure_df = (
    pl.from_pandas(pd_features_df[["UPRN"]].join(results_df, how="left"))
    .with_columns(
        tenure_prob=pl.concat_list(
            "owner-occupied", "rental (private)", "rental (social)"
        ).list.max(),
    )
    .drop(["owner-occupied", "rental (private)", "rental (social)"])
)

features_df = features_df.join(predicted_tenure_df, how="left", on="UPRN")

# %% [markdown]
# ## Add data - IMD decile

# %%
# Load IMD deciles per LSOA and load LSOA names
# Source: https://opendatacommunities.org/data/societal-wellbeing/imd2019/indices
imd_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/societal-wellbeing_imd2019_indices.csv",
    skip_rows=7,
)
# Source: https://geoportal.statistics.gov.uk/datasets/9dd1fe173d894b6a838b5ee016d3037e_0/about
lsoa_names_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/Lower_Layer_Super_Output_Area_(2021)_to_LAD_(April_2023)_Lookup_in_England_and_Wales.csv"
)

# %%
# Join IMD decile to UPRNs
imd_df = (
    imd_df.select(["Reference area", "2019"])
    .rename({"Reference area": "lsoa_name", "2019": "imd_decile"})
    .join(
        lsoa_names_df.select(["LSOA21CD", "LSOA21NM"]),
        how="left",
        left_on="lsoa_name",
        right_on="LSOA21NM",
    )
)

features_df = features_df.join(
    imd_df.select(["LSOA21CD", "imd_decile"]), how="left", on="LSOA21CD"
)

# %% [markdown]
# ## Fill missing data - garden size
#
# - use average by property type (+ area)
# - sum up the different garden sizes

# %%
# Load land registry polygons for Plymouth
# Source: https://use-land-property-data.service.gov.uk/datasets/inspire/download#local-authorities-for-P
land_extent_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/Land_Registry_Cadastral_Parcels.gml",
    columns=["NATIONALCADASTRALREFERENCE", "geometry"],
)

# Preprocess land polygons
land_extent_gdf = geo_utils.transform_gdf_drop_duplicates(land_extent_gdf)
land_extent_gdf["land_area_m2"] = land_extent_gdf["geometry"].area

# Get Plymouth OS building footprints
plymouth_building_footprints_gdf = (
    plymouth_residential_buildings_gdf[["geometry", "ID", "building_area_m2"]]
    .rename(columns={"ID": "building_id"})
    .assign(source="os")
    .copy()
)

# %%
# Remove land extent geometries which are not polygons
print(land_extent_gdf["geometry"].geom_type.value_counts())
land_extent_gdf = land_extent_gdf[land_extent_gdf.geometry.geom_type == "Polygon"]

# Get intersection of land polygons and building polygons
intersection_gdf = garden_size.generate_gdf_land_building_overlay(
    land_parcels_gdf=land_extent_gdf,
    building_footprints_gdf=plymouth_building_footprints_gdf,
)

# Get garden size
gardens_gdf = garden_size.generate_gdf_garden_size(intersection_gdf, land_extent_gdf)

# Join garden size to UPRNs
uprns_with_garden_gdf = pl.from_pandas(
    gpd.sjoin(
        os_uprn_gdf,
        gardens_gdf,
        how="inner",
        predicate="intersects",
    ).drop(columns=["geometry", "index_right"])
)

# Identify UPRNs with multiple garden sizes and find the range between min and max garden size calculated for them
duplicated_gdf = (
    uprns_with_garden_gdf.filter(pl.col("UPRN").is_duplicated())
    .group_by("UPRN")
    .agg(pl.min("garden_area_m2").alias("min"), pl.max("garden_area_m2").alias("max"))
    .with_columns((pl.col("max") - pl.col("min")).alias("range"))
)

# Deduplicate UPRNs to the smallest garden size - the join allows us to retain the other columns in the df
# TODO - use pl.over instead as it's more efficient
uprns_with_garden_gdf = uprns_with_garden_gdf.join(
    uprns_with_garden_gdf.group_by("UPRN").agg(pl.min("garden_area_m2")),
    how="left",
    on="UPRN",
).filter(pl.col("garden_area_m2") == pl.col("garden_area_m2_right"))

features_df = features_df.join(
    uprns_with_garden_gdf.select(
        ["UPRN", "NATIONALCADASTRALREFERENCE", "garden_area_m2"]
    ),
    how="left",
    on="UPRN",
)

# %%
# Plot the distribution of range in garden size for UPRNs joined to multiple garden sizes
plt.boxplot(duplicated_gdf.filter(pl.col("range") <= 250000)["range"])
plt.title("Range in garden size (under 250,000m2) for duplicate UPRNs")
plt.ylabel("Range in garden area (m2)")
plt.show()

print(duplicated_gdf["range"].describe())

# %%
# Fill any remaining missing garden size with the median size per property type and OA
fill_garden_df = features_df.group_by(["predicted_property_type", "OA21CD"]).agg(
    pl.median("garden_area_m2").alias("fill_missing_garden_area_m2")
)
features_df = features_df.join(
    fill_garden_df, how="left", on=["predicted_property_type", "OA21CD"]
).with_columns(
    pl.col("garden_area_m2")
    .fill_null(pl.col("fill_missing_garden_area_m2"))
    .alias("use_garden_area_m2")
)

# %% [markdown]
# ## Communal heating
# - if any other UPRNs in the building have communal label -> communal
# - elif any other UPRNs in the building have individual -> individual
# - else null

# %%
# Add communal heating column (uses same logic as flats analysis)
features_df = fuel_type.extend_df_communal_heating(
    features_df, epc_col="MAIN_FUEL", name="community_heating"
)
features_df = fuel_type.extend_df_communal_heating(
    features_df, epc_col="MAINHEAT_DESCRIPTION", name="fill_community_heating"
)
features_df = features_df.with_columns(
    (pl.col("community_heating").fill_null(pl.col("fill_community_heating"))).alias(
        "community_heating"
    )
).drop("fill_community_heating")

# Map building IDs to UPRNs and join to features df
plymouth_uprn_to_building_mapping_df = os_openmap_buildings_plymouth_gdf.sjoin(
    plymouth_residential_uprns_gdf, how="inner", predicate="contains"
)
features_df = features_df.join(
    pl.from_pandas(plymouth_uprn_to_building_mapping_df[["ID", "UPRN"]]).rename(
        {"ID": "building_id"}
    ),
    how="left",
    on="UPRN",
)

# Identify buildings that contain UPRNs with communal heating and buildings that contain UPRNs with individual heating
communal_heating_buildings = list(
    features_df.filter(pl.col("community_heating"))["building_id"].unique()
)
individual_heating_buildings = list(
    features_df.filter(
        (~pl.col("community_heating")) & pl.col("community_heating").is_not_null()
    )["building_id"].unique()
)
len(set(communal_heating_buildings).intersection(set(individual_heating_buildings)))

# Apply communal heating rules outlined above
features_df = features_df.with_columns(
    pl.when(pl.col("building_id").is_in(communal_heating_buildings))
    .then(True)
    .when(pl.col("building_id").is_in(individual_heating_buildings))
    .then(False)
    .otherwise(None)
    .alias("fill_community_heating")
).with_columns(
    pl.col("community_heating")
    .fill_null(pl.col("fill_community_heating"))
    .alias("use_community_heating")
)

# %% [markdown]
# ## Final feature processing

# %%
# Fill nulls with predicted types
features_df = features_df.with_columns(
    pl.col("property_type")
    .fill_null(pl.col("predicted_property_type"))
    .alias("use_property_type"),
    pl.col("TENURE").fill_null(pl.col("predicted_tenure")).alias("use_tenure"),
)

# %% [markdown]
# ## Cluster UPRNs

# %%
# Cluster UPRNs on distance only
model = HDBSCAN(
    min_cluster_size=5,
    cluster_selection_epsilon=20,
    algorithm="ball_tree",
    metric="euclidean",
)
plymouth_residential_uprns_gdf["cluster"] = model.fit_predict(
    plymouth_residential_uprns_gdf[["X_COORDINATE", "Y_COORDINATE"]]
)
print(plymouth_residential_uprns_gdf["cluster"].nunique())

# Join clusters to features df
features_df = features_df.join(
    pl.from_pandas(plymouth_residential_uprns_gdf[["UPRN", "cluster"]]),
    how="left",
    on="UPRN",
)

# Add cluster size
features_df = features_df.with_columns(
    cluster_size=pl.col("UPRN").n_unique().over("cluster")
)

# %%
# PLOT CLUSTERS FOR STOKE WARD
lsoa_boundaries = get_datasets.load_gdf_ons_lsoa_bounds()
stoke_ward_lsoas = [
    "E01015155",
    "E01015153",
    "E01015172",
    "E01015047",
    "E01015151",
    "E01015173",
    "E01015051",
    "E01015169",  # used in example
    "E01015171",
    "E01015170",
    "E01015099",
    "E01015174",
    "E01015167",
    "E01015045",
    "E01015044",
    "E01015168",
    "E01015072",
    "E01015042",
]

# Get boundary for Stoke ward
stoke_ward = (
    lsoa_boundaries[lsoa_boundaries["LSOA21CD"].isin(stoke_ward_lsoas)]
    .dissolve()["geometry"]
    .values[0]
)

# Convert UPRN CRS to 4326 (WGS 84) for plotting and filter to Stoke ward only
wgs84_uprns_df = plymouth_residential_uprns_gdf[
    plymouth_residential_uprns_gdf[
        ["UPRN", "cluster", "LATITUDE", "LONGITUDE", "geometry"]
    ].within(stoke_ward)
].to_crs(epsg=4326)

# Convert building footprint CRS to 4326 (WGS 84) for plotting and filter to Stoke ward only
wgs84_buildings_df = plymouth_residential_buildings_gdf[
    plymouth_residential_buildings_gdf.within(stoke_ward)
].to_crs(epsg=4326)

# Get centroid of all UPRNs to locate the centre of the map
x, y = (
    wgs84_uprns_df.dissolve().centroid.values[0].x,
    wgs84_uprns_df.dissolve().centroid.values[0].y,
)

import random

# Get list of colours for plotting clusters
get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n + 1)]
colors = get_colors(wgs84_uprns_df["cluster"].max())

# Jitter latitude and longitude so we can see overlapping points when plotting
sigma = 0.00001
wgs84_uprns_df["jitter_lat"] = wgs84_uprns_df["LATITUDE"].apply(
    lambda x: np.random.normal(x, sigma)
)
wgs84_uprns_df["jitter_long"] = wgs84_uprns_df["LONGITUDE"].apply(
    lambda x: np.random.normal(x, sigma)
)

# Plot building footprints
map = folium.Map(location=[y, x], tiles="OpenStreetMap", zoom_start=15)
for _, r in wgs84_buildings_df.iterrows():
    geo_j = gpd.GeoSeries(r["geometry"]).to_json()
    geo_j = folium.GeoJson(
        data=geo_j, style_function=lambda x: {"fillColor": "orange", "color": "orange"}
    )
    geo_j.add_to(map)

# Plot jittered UPRN points and colour by cluster
for _, r in wgs84_uprns_df.iterrows():
    if r.cluster == -1:
        # Set noise points as grey and label as noise
        folium.Marker(
            location=[r["jitter_lat"], r["jitter_long"]],
            icon=folium.Icon(icon_color="grey", prefix="fa", icon="ban"),
            popup="Noise",
        ).add_to(map)
    else:
        # Create coloured circle markers for clustered UPRNs
        folium.CircleMarker(
            location=[r["jitter_lat"], r["jitter_long"]],
            radius=5,
            weight=5,
            alpha=0.5,
            color=colors[r.cluster],
            popup=f"Cluster {r.cluster}",
        ).add_to(map)
map

# %%
# SAVE DATA
select_features = [
    "UPRN",
    "cluster",
    "OA21CD",
    "LSOA21CD",
    "in_listed_building",
    "in_conservation_area",
    "in_heat_network_zone",
    "filled_off_gas",
    "use_property_type",
    "use_tenure",
    #  'tenure_prob',
    "imd_decile",
    "NATIONALCADASTRALREFERENCE",
    "use_garden_area_m2",
    "building_id",
    "use_community_heating",
]

rename_selected = {
    "OA21CD": "oa21",
    "use_garden_area_m2": "garden_area_m2",
    "use_property_type": "predicted_property_type",
    "use_tenure": "predicted_tenure",
    "filled_off_gas": "use_off_gas",
    "use_community_heating": "on_communal_heating",
}

save_utils.save_to_s3(
    features_df,
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/results/plymouth_features_full_with_clusters.parquet",
)
save_utils.save_to_s3(
    features_df.select(select_features).rename(rename_selected),
    "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/results/plymouth_features_selected_with_clusters.parquet",
)
plymouth_residential_uprns_gdf.to_file("plymouth_residential_uprns_with_clusters.shp")
