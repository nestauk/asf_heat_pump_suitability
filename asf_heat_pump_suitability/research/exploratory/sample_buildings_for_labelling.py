# %% [markdown]
# ## Sampling buildings for manual labelling
# Here we take a random sample of ~1000 OS OpenMap Local building footprints from Plymouth and a number of other areas (listed below). The samples are taken from a subset of building footprints which contain more than one UPRN which is labelled as a flat. The building footprints will be manually labelled with participating labellers in Google MyMaps with codes representing their different flat archetypes. Ultimately, this labelled data will be used to train a model to classify buildings into blocks of flats and not.
#
# We started with a sample of 200 buildings in Plymouth, then added an additional 100. Finally, we are taking a sample of 700 from a mixture of Plymouth and the other towns/cities listed below.
# He = town/city with heterogeneous buildings; Ho = homoegeneous buildings
# - (South) Plymouth (He) - x700. 300 samples already. 400 left to sample.
# - (North) Nottingham (He) - x60
# - (North) Bradford (Ho - terraces) - x60
# - (Scotland) Glasgow (Ho - tenements) - x60
# - (North) Manchester (Ho - blocks) x60
# - (South) Bath (Ho - terraced) x60
#

# %%
import geopandas as gpd
import polars as pl
import numpy as np
import random

import simplekml
import boto3

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import load_boundaries, load_tree_input
from asf_heat_pump_suitability.pipeline.transform import uprns

from datetime import date
import os

today = date.today().strftime("%Y%m%d")

# %%
seed = 10
s3 = boto3.resource("s3")
BUCKET = "asf-heat-pump-suitability"

# %%
labellers = ["reindeer", "springbok", "anteater", "raccoon", "yak", "leopard"]

# %% [markdown]
# ## Compare April and October buildings files

# %%
# Load April 2025 building footprints
apr_buildings_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/inputs/v042025_OSOpenMapLocal_geometries_selected/SX/SX_Building.shp"
)

# Load October 2025 building footprints
oct_buildings_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/inputs/geodata/v202510_OSOpenMapLocal_geometries_selected/SX/SX_Building.shp"
)

# Print diffs
print("N building footprints in April 2025 dataset:")
print(len(apr_buildings_gdf))

print("\nN building footprints in October 2025 dataset:")
print(len(oct_buildings_gdf))

print("\nN extra building footprints in October 2025 dataset:")
print(len(oct_buildings_gdf) - len(apr_buildings_gdf))

# Normalize geometry columns to make sure the coordinates are in the same order
apr_buildings_gdf["geometry"] = apr_buildings_gdf.normalize()
oct_buildings_gdf["geometry"] = oct_buildings_gdf.normalize()

# Merge April and October on geometry
joined_buildings_gdf = apr_buildings_gdf.merge(
    oct_buildings_gdf, how="outer", on="geometry", suffixes=("_apr", "_oct")
)

# Get proportion of building footprints which persist across April and October 2025
persistent_buildings_gdf = joined_buildings_gdf[
    (~joined_buildings_gdf["ID_apr"].isna()) & (~joined_buildings_gdf["ID_oct"].isna())
]

print("\nCount of building footprints which persist:")
print(len(persistent_buildings_gdf))

print("\nProportion of building footprints which persist:")
print(len(persistent_buildings_gdf) / len(apr_buildings_gdf))

# %% [markdown]
# ## Original random samples of buildings containing flats from Plymouth
#
# See instructions to save polygons to kml with simplekml here: https://simplekml.readthedocs.io/en/latest/gettingstarted.html#creating-a-polygon

# %%
# Load Plymouth buildings and residential UPRNs with property type label
buildings_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/inputs/v042025_OSOpenMapLocal_geometries_selected/SX/SX_Building.shp"
)
uprns_df = pl.read_parquet(
    f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_flats.parquet"
)

# Join buildings to UPRNs
uprns_gdf = (
    uprns.generate_gdf_uprn_coords(df=uprns_df)
    .sjoin(buildings_gdf, how="inner", predicate="within")
    .drop(columns=["index_right", "FEATCODE"])
)
uprns_df = pl.from_pandas(uprns_gdf.drop(columns="geometry")).rename(
    {"ID": "building_id"}
)

# %%
# We need to assert there are only Polygons in the building footprints, otherwise taking the outerboundary is not sufficient
assert (buildings_gdf["geometry"].geom_type == "Polygon").all()

# %%
# Sample 1
n = 200
pl.set_random_seed(seed)

# Identify all buildings containing any flats
buildings_containing_flats = uprns_df.filter(pl.col("property_type_flat"))[
    "building_id"
].unique(maintain_order=True)

# Filter to buildings with flats, then get total count of flats and UPRNs per building
sample_df = (
    uprns_df.filter(
        pl.col("building_id").is_in(buildings_containing_flats),
    )
    .group_by("building_id")
    .agg(
        pl.col("property_type_flat").sum().alias("n_flats"),
        pl.col("UPRN").count().alias("n_total"),
        # Take a random sample from any building with >4 flats
        # This is based on an initial assumption that a block of flats should contain >4 flats
    )
    .filter(pl.col("n_flats") > 4)
    .sort(by="building_id")
    .sample(n=n, seed=seed)
)

# Merge sample buildings with their geometries
sample_gdf = buildings_gdf[["ID", "geometry"]].merge(
    sample_df.to_pandas(), how="inner", left_on="ID", right_on="building_id"
)
print(f"Sample length: {len(sample_gdf)}")

# Convert to 4326 projection for plotting on MyMaps
sample_4326_gdf = sample_gdf.to_crs(epsg=4326)
# Get centroid coordinates of each building to geolocate on google maps
sample_4326_gdf = sample_4326_gdf.merge(
    sample_4326_gdf.centroid.get_coordinates(),
    how="left",
    left_index=True,
    right_index=True,
)
# Create google maps link
sample_4326_gdf["url"] = sample_4326_gdf.apply(
    lambda row: f"https://www.google.com/maps/search/?api=1&query={row['y']},{row['x']}",
    axis=1,
)

# Save as kml file
kml = simplekml.Kml()
for idx, r in sample_4326_gdf.iterrows():
    pol = kml.newpolygon(
        name="unlabelled",
        description=f"Location: https://www.google.com/maps/search/?api=1&query={r['y']},{r['x']}\nN flats: {r['n_flats']}\nN total: {r['n_total']}",
        # Convert the exterior boundary of the polygon to a linear ring and get the coordinates
        outerboundaryis=list(r["geometry"].exterior.coords),
    )
    pol.style.polystyle.color = "9939FF14"
    # Set outline to True
    pol.style.polystyle.outline = 1

fname = (
    f"{today}_UNLABELLED_plymouth_buildings_containing_flats_sample_n{n}_seed{seed}.kml"
)
kml.save(fname)
file = os.path.join(os.getcwd(), fname)
s3.Bucket(BUCKET).upload_file(
    file, os.path.join("local_heat_planning", "labelling", fname)
)

shp_file = f'{fname.split(".")[0]}.shp'
sample_gdf.to_file(shp_file)
s3.Bucket(BUCKET).upload_file(
    os.path.join(os.getcwd(), shp_file),
    os.path.join("local_heat_planning", "labelling", shp_file),
)

# %%
# Sample 2 - take additional sample
# This sample selects from a different subset of buildings as sample 1
# The purpose is to try to capture a larger number of ambiguous buildings from the data
n = 100
pl.set_random_seed(seed)

buildings_containing_flats = uprns_df.filter(pl.col("property_type_flat"))[
    "building_id"
].unique(maintain_order=True)
already_labelled = sample_df["building_id"].unique()

# Sample buildings containing flats but only if they haven't been labelled
small_building_sample_df = (
    uprns_df.filter(
        pl.col("building_id").is_in(buildings_containing_flats),
        ~pl.col("building_id").is_in(already_labelled),
    )
    .group_by("building_id")
    .agg(
        pl.col("property_type_flat").sum().alias("n_flats"),
        pl.col("UPRN").count().alias("n_total"),
        # Sample from any buildings with >1 flat and <=15 UPRNs
    )
    .filter(pl.col("n_flats") > 1, pl.col("n_total") <= 15)
    .sort(by="building_id")
    .sample(n=n, seed=seed)
)

# Join building geometry
small_building_sample_gdf = buildings_gdf[["ID", "geometry"]].merge(
    small_building_sample_df.to_pandas(),
    how="inner",
    left_on="ID",
    right_on="building_id",
)
print(f"Sample length: {len(small_building_sample_gdf)}")

# Convert to 4326 projection and create google maps URL
small_building_sample_4326_gdf = small_building_sample_gdf.to_crs(epsg=4326)
small_building_sample_4326_gdf = small_building_sample_4326_gdf.merge(
    small_building_sample_4326_gdf.centroid.get_coordinates(),
    how="left",
    left_index=True,
    right_index=True,
)
small_building_sample_4326_gdf["url"] = small_building_sample_4326_gdf.apply(
    lambda row: f"https://www.google.com/maps/search/?api=1&query={row['y']},{row['x']}",
    axis=1,
)

# Save to kml file for MyMaps
kml = simplekml.Kml()
for idx, r in small_building_sample_4326_gdf.iterrows():
    pol = kml.newpolygon(
        name="unlabelled",
        description=f"Location: https://www.google.com/maps/search/?api=1&query={r['y']},{r['x']}\nN flats: {r['n_flats']}\nN total: {r['n_total']}",
        outerboundaryis=list(r["geometry"].exterior.coords),
    )
    pol.style.polystyle.color = "9939FF14"
    pol.style.polystyle.outline = 1

fname = f"{today}_UNLABELLED_small_plymouth_buildings_containing_flats_sample_n{n}_seed{seed}.kml"
kml.save(fname)
file = os.path.join(os.getcwd(), fname)
s3.Bucket(BUCKET).upload_file(
    file, os.path.join("local_heat_planning", "labelling", fname)
)

shp_file = f'{fname.split(".")[0]}.shp'
sample_gdf.to_file(shp_file)
s3.Bucket(BUCKET).upload_file(
    os.path.join(os.getcwd(), shp_file),
    os.path.join("local_heat_planning", "labelling", shp_file),
)

# %% [markdown]
# # Additional samples from Plymouth and other areas
#
# We are sampling mainly from Plymouth, but we want to sample from a variety of areas to do some initial testing of the robustness of our model across areas. We have selected a mix of areas which we think have heterogeneous (He) building archetypes, and more homogeneous (Ho) building archetypes. We have selected different towns/cities across the North and South of England, and Scotland.
#
# - (South) Plymouth (He) - x700. 300 samples already. 400 left to sample.
# - (North) Nottingham (He) - x60
# - (North) Bradford (Ho - terraces) - x60
# - (Scotland) Glasgow (Ho - tenements) - x60
# - (North) Manchester (Ho - blocks) x60
# - (South) Bath (Ho - terraced) x60
#
# Get count of flats per building footprint and group into 2-6 flats, 7-15 flats, and 16+ flats. Take 33% of the sample for each city from them.

# %%
# Set number of samples to take from each sampling area
n_samples = {k: 60 for k in config["constant"]["sampling_areas"]}
n_samples["Plymouth"] = 400

# %%
# LOAD REQUIRED DATA TO SAMPLE BUILDINGS FROM
# Load building footprints for sampling areas and load UPRNs
buildings_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="building", grid_squares=config["constant"]["grid_squares"]["sampling_areas"]
)
sampling_areas_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/sampling_areas_residential_uprns_with_flats.parquet"
)

# Join buildings to UPRNs
sampling_areas_gdf = (
    uprns.generate_gdf_uprn_coords(sampling_areas_df)
    .sjoin(buildings_gdf, how="inner", predicate="within")
    .drop(columns=["index_right", "FEATCODE"])
).drop_duplicates(subset="UPRN")
sampling_areas_df = pl.from_pandas(sampling_areas_gdf.drop(columns="geometry")).rename(
    {"ID": "building_id"}
)

# Add LA name to each UPRN
la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
    select_las=config["constant"]["sampling_areas"]
)
sampling_areas_gdf = sampling_areas_gdf.sjoin(
    la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
    how="left",
    predicate="within",
).drop(columns="index_right")

sampling_areas_df = pl.from_pandas(sampling_areas_gdf.drop(columns="geometry")).rename(
    {"ID": "building_id"}
)

# %%
# We need to assert there are only Polygons in the building footprints, otherwise taking the outerboundary is not sufficient
assert (buildings_gdf["geometry"].geom_type == "Polygon").all()

# %%
# CREATE SAMPLE OF BUILDINGS TO LABEL
pl.set_random_seed(seed)

# Identify already labelled buildings
already_labelled_apr_id = set(sample_df["building_id"].unique()) | set(
    small_building_sample_df["building_id"].unique()
)
apr_sample_gdf = apr_buildings_gdf[
    apr_buildings_gdf["ID"].isin(already_labelled_apr_id)
]
joined_sample_gdf = apr_sample_gdf.merge(
    oct_buildings_gdf, how="left", on="geometry", suffixes=["_apr", "_oct"]
)

print("N samples from April missing from October:")
print(joined_sample_gdf["ID_oct"].isna().sum())

already_labelled_apr_id = joined_sample_gdf["ID_apr"].tolist()
already_labelled_oct_id = joined_sample_gdf["ID_oct"].tolist()

apr_to_oct_id = dict(zip(already_labelled_apr_id, already_labelled_oct_id))
already_labelled_df = (
    pl.concat([sample_df, small_building_sample_df])
    .rename({"building_id": "building_id_apr"})
    .with_columns(pl.col("building_id_apr").replace(apr_to_oct_id).alias("building_id"))
    .select(["building_id", "n_flats", "n_total"])
)

# Prepare already labelled dataframe with same columns for later sampling
already_labelled_df = already_labelled_df.with_columns(
    # Setting LA name to Plymouth for all lines of already labelled data
    pl.Series("la_name", ["Plymouth"] * already_labelled_df.height),
    # Group number of flats
    pl.when(pl.col("n_flats").is_between(2, 6, closed="both"))
    .then(pl.lit("2-6_flats"))
    .when(pl.col("n_flats").is_between(7, 15, closed="both"))
    .then(pl.lit("7-15_flats"))
    .when(pl.col("n_flats") > 15)
    .then(pl.lit("16+_flats"))
    .otherwise(None)
    .alias("n_flats_grouped"),
    # Set labeller name to "reindeer" for all lines of labelled data
    pl.Series("labeller", ["reindeer"] * already_labelled_df.height),
)

# Get buildings containing flats
buildings_containing_flats = sampling_areas_df.filter(pl.col("property_type_flat"))[
    "building_id"
].unique(maintain_order=True)

# Create a dataframe of potential buildings data for labelling by
# 1) Removing buildings that have already been sampled & labelled
# 2) Enhancing dataframe with info such as: local authority where building is located, number of flats in the building and additional variables
sample_from_df = (
    sampling_areas_df.filter(
        pl.col("building_id").is_in(buildings_containing_flats),
        ~pl.col("building_id").is_in(already_labelled_oct_id),
    )
    .group_by("building_id")
    .agg(
        pl.col("property_type_flat").sum().alias("n_flats"),
        pl.col("UPRN").count().alias("n_total"),
        pl.col("LAD23NM").first().alias("la_name"),
        # Sample from any buildings with >1 flat
    )
    .filter(pl.col("n_flats") > 1)
    .with_columns(
        # Group number of flats
        pl.when(pl.col("n_flats").is_between(2, 6, closed="both"))
        .then(pl.lit("2-6_flats"))
        .when(pl.col("n_flats").is_between(7, 15, closed="both"))
        .then(pl.lit("7-15_flats"))
        .when(pl.col("n_flats") > 15)
        .then(pl.lit("16+_flats"))
        .otherwise(None)
        .alias("n_flats_grouped")
    )
)

samples = []

n_flats_groups = ["2-6_flats", "7-15_flats", "16+_flats"]

# Take samples for each Local Authority for each group of flat counts to create sample of 700 buildings
for la, df in sample_from_df.sort(by="la_name").group_by(
    "la_name", maintain_order=True
):
    # Get the number of samples per LA
    n = n_samples[la[0]]
    # Take one third of samples from each group of flat counts
    for n_flats in n_flats_groups:
        # proportion of samples from each group of flat counts, in this case it's one third because there are 3 groups
        _n = np.ceil(n / len(n_flats_groups))
        samples.append(
            df.filter(pl.col("n_flats_grouped") == n_flats)
            .sort(by="building_id")
            .sample(n=_n, seed=seed)
        )

sample_3_df = pl.concat(samples)

# %%
# ASSIGN LABELLERS TO SAMPLE
# Get number of samples per labeller
n_samples_per_labeller = int(np.ceil(sample_3_df.height / len(labellers)))
# Create list of labellers corresponding to total sample length
labeller_assignments = labellers * n_samples_per_labeller

# Shuffle the list for random assignment
random.seed(seed)
random.shuffle(labeller_assignments)

# Assign a labeller to each row
# Take only the required number of labellers (the labeller_assignments list can be slightly too long due to rounding)
labeller_assignments = labeller_assignments[: sample_3_df.height]
sample_3_df = sample_3_df.with_columns(pl.Series("labeller", labeller_assignments))

second_samples = []
final_samples = {}

# Create the final subset for each labeller
for labeller in labellers:
    # Get the number of primary samples selected for the labeller
    n_samples = sample_3_df.filter(pl.col("labeller") == labeller).height

    # Calculate number of additional samples so that some buildings get 2 labels from different people
    total_sample = n_samples_per_labeller + 30
    additional_sample = total_sample - n_samples

    # Get the second sample
    if labeller != "reindeer":
        print(
            f"Labeller: {labeller}, adding reindeer's original sample for additional sampling/ double labelling..."
        )
        second_sample_from = pl.concat([sample_3_df, already_labelled_df])
        second_sample = (
            second_sample_from.filter(
                pl.col("labeller") != labeller,
                ~pl.col("building_id").is_in(second_samples),
            )
            .sort("building_id")
            .sample(additional_sample, seed=seed)
        )
    else:
        second_sample = (
            sample_3_df.filter(
                pl.col("labeller") != labeller,
                ~pl.col("building_id").is_in(second_samples),
            )
            .sort("building_id")
            .sample(additional_sample, seed=seed)
        )

    # Add the IDs of the second sample to the list so that they cannot be sampled again
    second_samples.extend(list(second_sample["building_id"]))

    # Concatenate the final sample for each labeller
    final_sample = pl.concat(
        [sample_3_df.filter(pl.col("labeller") == labeller), second_sample]
    )
    final_samples[labeller] = final_sample

    print(f"Labeller: {labeller}, N samples: {final_sample.height}")

# %%
# For each labeller, show the number of samples in your set for labelling that is being labelled by each other labeller
for l, sample in final_samples.items():
    print(l)
    print(sample["labeller"].value_counts())

# %%
# JOIN BUILDING GEOMETRY AND SAVE TO KML

for labeller, df in final_samples.items():
    # Join building geometry
    sample_3_gdf = buildings_gdf[["ID", "geometry"]].merge(
        df.to_pandas(), how="inner", left_on="ID", right_on="building_id"
    )
    l = len(sample_3_gdf)

    # Convert to 4326 projection and create google maps URL
    sample_3_4326_gdf = sample_3_gdf.to_crs(epsg=4326)
    sample_3_4326_gdf = sample_3_4326_gdf.merge(
        sample_3_4326_gdf.centroid.get_coordinates(),
        how="left",
        left_index=True,
        right_index=True,
    )
    sample_3_4326_gdf["url"] = sample_3_4326_gdf.apply(
        lambda row: f"https://www.google.com/maps/search/?api=1&query={row['y']},{row['x']}",
        axis=1,
    )

    # Save to kml file for MyMaps
    kml = simplekml.Kml()
    for idx, r in sample_3_4326_gdf.iterrows():
        pol = kml.newpolygon(
            name="unlabelled",
            description=f"Location: https://www.google.com/maps/search/?api=1&query={r['y']},{r['x']}\nN flats: {r['n_flats']}\nN total: {r['n_total']}",
            outerboundaryis=list(r["geometry"].exterior.coords),
        )
        pol.style.polystyle.color = "9939FF14"
        pol.style.polystyle.outline = 1

    kml.save(
        f"{today}_{labeller}_UNLABELLED_small_plymouth_buildings_containing_flats_sample_n{l}_seed{seed}.kml"
    )

files_to_upload = [
    f"{today}_{labeller}_UNLABELLED_small_plymouth_buildings_containing_flats_sample_n{l}_seed{seed}.kml"
    for labeller in labellers
]
for file in files_to_upload:
    s3.Bucket(BUCKET).upload_file(
        os.path.join(os.getcwd(), file),
        os.path.join("local_heat_planning", "labelling", file),
    )
