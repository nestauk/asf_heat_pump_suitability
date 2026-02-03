# %%
# %% [markdown]
# # Visualising the results of the decision tree for Greater Manchester
#
# In this notebook we:
# - Code the decision tree that outputs the most suitable tech for each property given the following inputs: wether it is in a city centre or planned HN zone, garden size and whether the property is in a block of flats or not.
# - Map the outputs of the decision tree for all domestic buildings in Greater Manchester

# %%
# package imports
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
import os

# %%
# local imports
from asf_heat_pump_suitability import PROJECT_DIR
from asf_heat_pump_suitability.pipeline.transform.uprns import generate_gdf_uprn_coords
from asf_heat_pump_suitability.getters.load_tree_input import (
    load_gdf_os_openmap_local_layer,
)
from asf_heat_pump_suitability.getters.load_geodata import (
    load_gdf_heat_network_zones,
)
from asf_heat_pump_suitability.getters import (
    load_boundaries,
)

# %%
colours = {
    "Individual solution": "#18A48C",
    "Networked GSHP": "#0000FF",
    "Communal solutions": "#FF6E47",
    "District heat network": "#EA2541",
    "Individual solution or Networked GSHP": "grey",
    "Individual solution or District heat network": "gray",
}


# %% [markdown]
# ## 1. Loading data

# %% [markdown]
# ### 1.1. Greater Manchester residential UPRNs data
#
# Greater Manchester residential UPRNs data contains info about:
# - each residential UPRN in Greater Manchester
# - X and Y coordinates of each UPRN (as well as latitude and longitude)

# %%
# Getting data and converting polars df to geodf
gm_uprns = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/greater_manchester_las_residential_uprns.parquet"
)
gm_uprns = generate_gdf_uprn_coords(df=gm_uprns)
gm_uprns.head()

# %% [markdown]
# ### 1.2. City centres and planned HNZ
#
# For each domestic UPRN we hvae:
# - X and Y coordinates of each UPRN
# - `in_city_centre`: a flag for whether it is located in a city centre according to a set of pre-defined spatial signature types (as per the Spatial Signatures Framework)
# - `spatial_signature_type`: the respective spatial signature type for each UPRN
# - `in_hn_zone`: a flag for whether it is located in a planned DESNZ heat network zone
#

# %%
# Getting data and converting polars df to geodf
hnz_city_centre_data = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/greater_manchester_las_residential_uprns_with_hn_zones_city_centres.parquet"
)
hnz_city_centre_data = generate_gdf_uprn_coords(df=hnz_city_centre_data)
hnz_city_centre_data.head()

# %%
hnz_city_centre_data

# %% [markdown]
# ### 1.3. Greater Manchester features dataset
#
# Greater Manchester features dataset will eventually contain all the features we need to run the decision tree for each domestic UPRN in Greater Manchester including garden size, blocks of flats flag, city centre flag and planned HNZ flag. For now, it only contains:
# - each residential UPRN in Greater Manchester
# - X and Y coordinates of each UPRN
# - wether property is a flat
# - NATIONALCADASTRALREFERENCE which identifies the specific building where the property is located
# - max contiguous and total outdoor space in m2

# %%
# Getting data and converting polars df to geodf
gm_with_features = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/greater_manchester_las_residential_uprns_with_features.parquet"
)
gm_with_features = generate_gdf_uprn_coords(df=gm_with_features)
gm_with_features.head()

# %% [markdown]
# ### 1.4. Building footprints
#
# Building footprints dataset contains the geometries of all buildings in grid squares including residential and non-residential buildings. This includes an area bigger than Greater Manchester boundary, so it needs to be filtered to the Greater Manchester boundary.
#
# It includes building footprint IDs and geometries of all buildings in grid squares To note that one building footprint ID sometimes contains multiple buildings.
#
# These are updated regularly by Ordnance Survey and with each update the building footprint IDs might change.

# %%
from asf_heat_pump_suitability import config

# Loading building footprints for grid squares
building_footprints = load_gdf_os_openmap_local_layer(
    layer="building",
    grid_squares=config["constant"]["greater_manchester_las"]["grid_squares"],
)
building_footprints.head()

# %% [markdown]
# ### 1.5. Greater Manchester boundaries

# %%
# Loading the Greater Manchester LAs boundaries
gm_la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
    select_las=config["constant"]["greater_manchester_las"]["la_names"]
)

# %%
gm_la_boundaries_gdf

# %% [markdown]
# ### 1.6. Planned HN zones geometries
#
# This dataset contains the geometries of all planned heat network zones in Greater Manchester as per DESNZ data.

# %%
gm_hn_zones_gdf = load_gdf_heat_network_zones(local_authority="greater_manchester_las")

# %%
gm_hn_zones_gdf.crs

# %%
gm_hn_zones_gdf.head()

# %% [markdown]
# ## 2. Processing data
#
# Most of the processing steps below are temporary as there will be pipelines that will put all of this data together into one Greater Manchester features dataset.

# %% [markdown]
# ### 2.1. [temporary] Joining all dfs into one geodf

# %%
gm_uprns.set_index("UPRN", inplace=True)
gm_with_features.set_index("UPRN", inplace=True)
hnz_city_centre_data.set_index("UPRN", inplace=True)

# %%
gm_uprns.head()

# %%
# checking if lengths are the same
len(gm_uprns), len(gm_with_features), len(hnz_city_centre_data)

# %%
# Joining all geodfs into one based on UPRN
gm_gdf = gm_uprns.join(hnz_city_centre_data[["in_city_centre", "in_hn_zone"]])
gm_gdf = gm_gdf.join(
    gm_with_features.drop(columns=["X_COORDINATE", "Y_COORDINATE", "geometry"])
)

# %%
gm_gdf.reset_index(inplace=True)

# %%
gm_gdf.head()

# %% [markdown]
# ### 2.2. Spatial join to add building footprints to main geodf
#
# Adding geometries of building footprints in Greater Manchester to the main geodf.

# %%
# we create a copy of the (building) geometry column so that we can keep it after the spatial join
gm_gdf["property_geometry"] = gm_gdf.geometry
building_footprints["building_geometry"] = building_footprints["geometry"]
gm_gdf = gm_gdf.sjoin(
    building_footprints[["geometry", "building_geometry"]],
    how="left",
    predicate="within",
).drop(columns=["index_right"])

# %% [markdown]
# ### 2.3. Spatial join with DESNZ HN zones geometries data
#
# For each property in a planned HN zone, we add the HN zone geometry.

# %%
# Adding a column to indicate if the property is within a DESNZ heat network zone
gm_hn_zones_gdf["desnz_hn_zone"] = True

# Creating a copy of the geometry column to keep after the spatial join
gm_hn_zones_gdf["desnz_hn_zone_geometry"] = gm_hn_zones_gdf["geometry"]

# %%
gm_gdf = gm_gdf.sjoin(
    gm_hn_zones_gdf[["geometry", "desnz_hn_zone_geometry"]],
    how="left",
    predicate="within",
)

# %% [markdown]
# ### 2.4. Load blocks of flats model, create features and create blocks of flats flag

# %%
# Loading blocks of flats model
import pickle
import boto3

s3client = boto3.client(
    "s3",
)
s3_bucket = "asf-heat-pump-suitability"
response = s3client.get_object(
    Bucket=s3_bucket, Key="local_heat_planning/outputs/blocks_of_flats_model.pkl"
)

body = response["Body"].read()
block_of_flats_model = pickle.loads(body)


# %%
# Create features to apply blocks of flats model

# %%

# Create features from building data
from asf_heat_pump_suitability.pipeline.transform import uprns

# pandas to polars
df = pl.from_pandas(
    gm_gdf.drop(
        columns=[
            "geometry",
            "property_geometry",
            "building_geometry",
            "desnz_hn_zone_geometry",
            "index_right",
        ]
    )
)

buildings_w_uprns_gdf = building_footprints.sjoin(
    uprns.generate_gdf_uprn_coords(df=df),
    how="left",
    predicate="contains",
).drop(columns=["index_right"])

buildings_w_uprns_gdf = buildings_w_uprns_gdf[~buildings_w_uprns_gdf["UPRN"].isna()]
buildings_w_uprns_gdf["building_area_m2"] = buildings_w_uprns_gdf.area
buildings_w_uprns_gdf["building_perimeter_m"] = buildings_w_uprns_gdf.length
buildings_w_uprns_df = pl.from_pandas(
    buildings_w_uprns_gdf.drop(columns=["geometry", "building_geometry"])
)

agg_building_df = (
    buildings_w_uprns_df.group_by("ID")
    .agg(
        pl.col("UPRN").count().alias("n_UPRNs"),
        pl.col("property_type_flat").sum().alias("n_flats"),
        pl.col("building_area_m2").first().name.keep(),
        pl.col("building_perimeter_m").first().name.keep(),
    )
    .filter(pl.col("n_flats") > 0)
    .with_columns(
        (pl.col("n_flats") / pl.col("n_UPRNs")).alias("proportion_flats"),
        (pl.col("n_UPRNs") / pl.col("building_area_m2")).alias("UPRNs_per_building_m2"),
    )
)


# %%

uprns_gdf = (
    uprns.generate_gdf_uprn_coords(df=df)
    .sjoin(building_footprints, how="inner", predicate="within")
    .drop(columns=["index_right"])
)


# Create concave hull feature to represent spatial distribution of UPRNs in each building
hull_gdf = uprns_gdf.dissolve("ID").concave_hull().reset_index()


hull_gdf = hull_gdf.rename(columns={0: "geometry"}).set_geometry("geometry")
hull_gdf["concave_hull_area_m2"] = hull_gdf.area

# Join building features with concave hull feature
agg_building_df = agg_building_df.join(
    pl.from_pandas(hull_gdf.drop(columns="geometry")),
    how="left",
    on="ID",
)

# Calculate additional features from the concave hull area
agg_building_df = agg_building_df.with_columns(
    (pl.col("n_UPRNs") / pl.col("concave_hull_area_m2")).alias(
        "uprns_per_hull_area_m2"
    ),
    (pl.col("n_flats") / pl.col("concave_hull_area_m2")).alias(
        "flats_per_hull_area_m2"
    ),
).with_columns(
    # UPRNs or flats per hull area can be infinite if all UPRNs/flats share the same coordinates (i.e. area = 0m2)
    # We change this to -1 for the model
    pl.when(pl.col("uprns_per_hull_area_m2").is_infinite())
    .then(-1)
    .otherwise(pl.col("uprns_per_hull_area_m2"))
    .alias("uprns_per_hull_area_m2"),
    pl.when(pl.col("flats_per_hull_area_m2").is_infinite())
    .then(-1)
    .otherwise(pl.col("flats_per_hull_area_m2"))
    .alias("flats_per_hull_area_m2"),
)

# Get count of UPRNs at each X and Y coordinates to get the count of UPRNs which share an exact location
uprns_df = pl.from_pandas(uprns_gdf.drop(columns=["geometry", "building_geometry"]))

# Get count of UPRNs at each X and Y coordinates to get the count of UPRNs which share an exact location
uprns_gdf = uprns_df.with_columns(
    n_stacked_uprns=pl.col("UPRN").count().over(["X_COORDINATE", "Y_COORDINATE"])
)


# Group by building and get the average and STD of UPRNs sharing the same coordinates
agg_uprns_df = uprns_gdf.group_by("ID").agg(
    pl.col("n_stacked_uprns").mean().alias("avg_n_stacked_uprns"),
    pl.col("n_stacked_uprns").std().alias("std_n_stacked_uprns"),
)

# Join all the calculated features together
features_df = agg_building_df.join(
    agg_uprns_df.select(["ID", "avg_n_stacked_uprns", "std_n_stacked_uprns"]),
    how="left",
    on="ID",
)


# %%
# These are the model features
features = [
    "n_UPRNs",
    "n_flats",
    "building_area_m2",
    "building_perimeter_m",
    "proportion_flats",
    "UPRNs_per_building_m2",
    "concave_hull_area_m2",
    "uprns_per_hull_area_m2",
    "flats_per_hull_area_m2",
    "avg_n_stacked_uprns",
    "std_n_stacked_uprns",
]

# %%
# apply model to create blocks of flats flag
features_df = features_df.to_pandas()
features_df["block_of_flats"] = block_of_flats_model.predict(features_df[features])

# %%
building_predictions_df = features_df[["ID", "block_of_flats"]]
building_predictions_df = pl.from_pandas(building_predictions_df)

# %%
uprn_predictions = (
    uprns_df.drop_nulls(subset="ID")
    .select(["UPRN", "ID"])
    .join(
        building_predictions_df.select(
            [
                "ID",
                "block_of_flats",
            ]
        ),
        how="left",
        on="ID",
    )
)

# %%
# uprn_predictions = uprn_predictions.with_columns(
#     pl.col("block_of_flats").fill_null(False)
# )

uprn_predictions = uprn_predictions.to_pandas()

# %%
# Join blocks of flats model data to gm_gdf
gm_gdf = gm_gdf.join(
    uprn_predictions.rename(columns={"block_of_flats": "in_block_of_flats"})[
        ["in_block_of_flats"]
    ],
    how="left",
)

# %%
gm_gdf["in_block_of_flats"] = gm_gdf["in_block_of_flats"].fillna(False)

# %%
gm_gdf["in_city_centre"].value_counts(dropna=False)

# %%
gm_gdf["in_hn_zone"].value_counts(dropna=False)

# %% [markdown]
# ## 3. Defining decision tree and identifying 1st and 2nd most suitable solutions
#
# 2nd most suitable solution is the closest solution in the decision tree.


# %%
def define_decision_tree(
    in_block_of_flats: bool, garden_size: float, city_centre_or_hnz: bool
) -> dict:
    """
    Defines the decision tree to identify:
    - first and second most suitable low carbon heating solutions for each property.
    - the path taken in the decision tree.

    Args:
        in_block_of_flats (bool): Whether the property is in a block of flats.
        garden_size (float): Size of the garden in square meters.
        city_centre_or_hnz (bool): Whether the property is in the city centre or in a planned heat network zone.

    Returns:
        dict: A dictionary with the first and second most suitable heating solutions and the path taken in the decision tree.
    """

    if in_block_of_flats:
        if city_centre_or_hnz:
            return {
                1: "District heat network",
                2: "Communal solutions",
                "path": "1. blocks of flats and city centre",
            }
        else:
            return {
                1: "Communal solutions",
                2: "Networked GSHP",
                "path": "2. blocks of flats, not city centre",
            }
    else:
        if city_centre_or_hnz:
            if pd.isnull(garden_size):
                return {
                    1: "Individual solution or District heat network",
                    2: "Individual solution or District heat network",
                    "path": "Unknown garden size in city centre",
                }
            elif garden_size > 70:
                return {
                    1: "Individual solution",
                    2: "District heat network",
                    "path": "3. not blocks of flats, city centre, big garden (70m2)",
                }
            else:
                return {
                    1: "District heat network",
                    2: "Networked GSHP",
                    "path": "4. not blocks of flats, city centre, small or no garden",
                }
        else:
            if pd.isnull(garden_size):
                return {
                    1: "Individual solution or Networked GSHP",
                    2: "Networked GSHP or Communal solutions",
                    "path": "Unknown garden size not in city centre",
                }
            elif garden_size > 30:
                return {
                    1: "Individual solution",
                    2: "Networked GSHP",
                    "path": "5. not blocks of flats, not city centre, big garden (30m2)",
                }
            else:
                return {
                    1: "Networked GSHP",
                    2: "Communal solutions",
                    "path": "6. not blocks of flats, not city centre, small/no garden",
                }


# %%
gm_gdf["in_city_centre_or_hn_zone"] = gm_gdf["in_city_centre"] | gm_gdf["in_hn_zone"]

gm_gdf["most_suitable_solutions"] = gm_gdf.apply(
    lambda x: define_decision_tree(
        x["in_block_of_flats"],
        x["max_contiguous_outdoor_space_area_m2"],
        x["in_city_centre_or_hn_zone"],
    ),
    axis=1,
)

# %%
gm_gdf["1st_most_suitable_solution"] = gm_gdf["most_suitable_solutions"].apply(
    lambda x: x[1]
)
gm_gdf["2nd_most_suitable_solution"] = gm_gdf["most_suitable_solutions"].apply(
    lambda x: x[2]
)
gm_gdf["decision_tree_path"] = gm_gdf["most_suitable_solutions"].apply(
    lambda x: x["path"]
)

# %%
gm_gdf["decision_tree_path"].value_counts(normalize=True)

# %%
gm_gdf["1st_most_suitable_solution"].value_counts(normalize=True)

# %%
gm_gdf["1st_most_suitable_solution"].value_counts(dropna=False)

# %% [markdown]
# ## 4. Number of solutions per land parcel and building footprint
#
# In this section we check wether there are multiple solutions for properties located in the same land parcel and building footprint. Ideally, all properties in the same land parcel & footprint should have the same solution.
#
# As we can observe below, most land parcels and building footprints have one solution for all properties, but there are some buildings with multiple solutions.

# %%
# Distribution of unique 1st most suitable solutions per land parcel (in counts)
gm_gdf.groupby("NATIONALCADASTRALREFERENCE")[["1st_most_suitable_solution"]].nunique()[
    "1st_most_suitable_solution"
].value_counts()

# %%
# Distribution of unique 1st most suitable solutions per land parcel (in proportions)
gm_gdf.groupby("NATIONALCADASTRALREFERENCE")[["1st_most_suitable_solution"]].nunique()[
    "1st_most_suitable_solution"
].value_counts(normalize=True)

# %%
# Number of unique land parcels in Greater Manchester vs. number of buildings
# This shows that lots of land parcels are aggregated into one single building geometry/ footprint
gm_gdf["NATIONALCADASTRALREFERENCE"].nunique(), gm_gdf["building_geometry"].nunique()

# %%
# Identifying pairs of different 1st most suitable solutions per land parcel
solutions_per_land_parcel = gm_gdf.groupby("NATIONALCADASTRALREFERENCE")[
    "1st_most_suitable_solution"
].apply(set)
solutions_per_land_parcel = pd.DataFrame(solutions_per_land_parcel)
solutions_per_land_parcel["n_solutions"] = solutions_per_land_parcel[
    "1st_most_suitable_solution"
].apply(len)
solutions_per_land_parcel = solutions_per_land_parcel[
    solutions_per_land_parcel["n_solutions"] > 1
]
solutions_per_land_parcel.reset_index(inplace=True)
solutions_per_land_parcel["1st_most_suitable_solution_str"] = solutions_per_land_parcel[
    "1st_most_suitable_solution"
].apply(lambda x: ", ".join(x))
solutions_per_land_parcel.groupby("1st_most_suitable_solution_str")[
    ["NATIONALCADASTRALREFERENCE"]
].nunique()


# %%
# Identifying pairs of different 1st most suitable solutions per building footprint
solutions_per_footprint = gm_gdf.groupby("building_geometry")[
    "1st_most_suitable_solution"
].apply(set)
solutions_per_footprint = pd.DataFrame(solutions_per_footprint)
solutions_per_footprint["n_solutions"] = solutions_per_footprint[
    "1st_most_suitable_solution"
].apply(len)
solutions_per_footprint = solutions_per_footprint[
    solutions_per_footprint["n_solutions"] > 1
]
solutions_per_footprint.reset_index(inplace=True)
solutions_per_footprint["1st_most_suitable_solution_str"] = solutions_per_footprint[
    "1st_most_suitable_solution"
].apply(lambda x: ", ".join(x))
solutions_per_footprint.groupby("1st_most_suitable_solution_str")[
    ["building_geometry"]
].nunique()


# %%
# check if each land parcel appears only within one building footprint
land_parcel_to_footprint = gm_gdf.groupby("NATIONALCADASTRALREFERENCE")[
    "building_geometry"
].apply(set)
land_parcel_to_footprint = pd.DataFrame(land_parcel_to_footprint)
land_parcel_to_footprint["n_footprints"] = land_parcel_to_footprint[
    "building_geometry"
].apply(len)

# %%
# It doesn't seem to be the case
land_parcel_to_footprint["n_footprints"].value_counts()

# %%
land_parcel_to_footprint["n_footprints"].value_counts(normalize=True) * 100

# %% [markdown]
# In most cases:
# - the same land parcel should be in one building footprint only (hopefully?)
# - properties in the same land parcel should have the same most suitable solution
# - properties in the same building footprint should have the same most suitable solution

# %%
# This is currently based on Greater Manchester only (both pairs found in land parcels and building footprints)
# but needs to be generalised to all possible cases


def assign_unique_sol(solution_set: set) -> str:
    """
    Assigns a unique solution based on the combination of solutions in the set.

    Args:
        solution_set (set): A set of solutions.

    Returns:
        str: A unique solution assigned based on the combination.
    """
    if "District heat network" in solution_set and "Networked GSHP" in solution_set:
        return "Communal solutions"
    elif "District heat network" in solution_set:
        return "District heat network"
    elif "Networked GSHP" in solution_set:
        return "Networked GSHP"
    elif "Communal solutions" in solution_set:
        return "Communal solutions"
    elif "Individual solution or Networked GSHP" in solution_set:
        return "Individual solution or Networked GSHP"
    elif "Individual solution or District heat network" in solution_set:
        return "Individual solution or District heat network"
    else:
        print("This shouldn't happen!")


solutions_per_footprint["final_solution"] = solutions_per_footprint[
    "1st_most_suitable_solution"
].apply(lambda x: assign_unique_sol(x))

# %%

solutions_per_footprint["final_solution"].value_counts(dropna=False)

# %%
mapping_set_to_final_solution = solutions_per_footprint.set_index("building_geometry")[
    "final_solution"
].to_dict()

# %%
gm_gdf["1st_most_suitable_solution"] = gm_gdf.apply(
    lambda x: (
        mapping_set_to_final_solution[x["building_geometry"]]
        if x["building_geometry"] in mapping_set_to_final_solution
        else x["1st_most_suitable_solution"]
    ),
    axis=1,
)

# %%
gm_gdf["1st_most_suitable_solution"].value_counts(dropna=False)


# %%
def map_blocks_of_flats_prob(
    tech: str,
    ward_df: gpd.GeoDataFrame,
    specific_ward_gdf: gpd.GeoDataFrame,
    labelled_tech: gpd.GeoDataFrame,
    colours: dict,
    threshold=None,
):
    """
    Maps properties suitable for a specific technology where the colour indicates the confidence in the block of flats label.

    Args:
        tech (str): a low carbon heating solution in the set {"Individual solution", "Networked GSHP", "Communal solutions", "District heat network"}
        ward_df (gpd.GeoDataFrame): ward data with most suitable solutions
        specific_ward_gdf (gpd.GeoDataFrame): specific ward boundary data
    """

    if tech != "":
        tech_specific_df = ward_df[ward_df["1st_most_suitable_solution"] == tech]
    else:
        tech_specific_df = ward_df.copy()
    fig, ax = plt.subplots(figsize=(15, 8))

    if threshold:
        tech_specific_df = tech_specific_df[
            tech_specific_df["block_of_flats_label_proba"] <= threshold
        ]

    print("tech:", tech)

    # mapping polygons of labelled data
    if tech != "District heat network":
        labelled_tech_specifc_df = labelled_tech[labelled_tech["Name"] == tech]
        labelled_tech_specifc_df.plot(
            ax=ax,
            column="Name",
            categorical=True,
            legend=True,
            color=labelled_tech_specifc_df["Name"].map(colours),
            edgecolor="black",
            linestyle="--",
            alpha=0.5,
        )
    else:
        labelled_tech.plot(
            ax=ax,
            color=colours[tech],
            edgecolor="black",
            linestyle="--",
            alpha=0.5,
        )

    greys = plt.get_cmap("Greys")
    import matplotlib.colors as mcolors

    # Create a new map using only the range from 0.2 (light gray) to 1.0 (black)
    # 0.0 would be white, so we skip it.
    cmap_no_white = mcolors.LinearSegmentedColormap.from_list(
        "trunc_greys", greys(np.linspace(0.2, 1.0, 256))
    )

    # mapping individual properties where the colour indicates probability of being a block of flats
    tech_specific_df.plot(
        ax=ax,
        column="block_of_flats_label_proba",
        cmap=cmap_no_white,
        legend=True,
        markersize=2,
        vmin=0.5,
        vmax=1,
    )

    # mapping ward boundary
    specific_ward_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

    # title and legend
    ax.set_title(
        f"Confidence in the blocks of flats label\n(black = very confident; light grey = lower confidence)",
        fontsize=12,
    )


# %%
fig, ax = plt.subplots(figsize=(15, 8))
gm_hn_zones_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1)
gm_la_boundaries_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)


# %%
def map_building_techs_in_ward(techs_gdf, col, specific_ward_gdf, colours, ward, map_):
    """
    Maps buildings in a specific ward with a predicted/labelled low carbon heating solutions.

    Args:
        techs_gdf (gpd.GeoDataFrame): buildings with most suitable solutions
        col (str): column with tech solution
        specific_ward_gdf (gpd.GeoDataFrame): specific ward boundary data
        colours (dict): mapping of technologies to colours
        ward (str): name of the ward
        map_ (str): takes ["Labelled", "Predicted"]
    """

    fig, ax = plt.subplots(figsize=(15, 8))

    techs_gdf.plot(
        ax=ax,
        column=col,
        categorical=True,
        legend=True,
        color=techs_gdf[col].map(colours),
        markersize=2,
    )

    # mapping ward boundary
    specific_ward_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

    # title and legend
    ax.set_title(f"{map_} most suitable solution for {ward}", fontsize=12)
    handles = []
    handles = [
        mpatches.Patch(facecolor=colours[tech], label=tech) for tech in colours.keys()
    ]
    ax.legend(handles=handles, loc="upper right")


# %%


# %% [markdown]
# ## 5. Visualising results per building in Greater Manchester

# %%
# Estimated most suitable
gm_building_most_suitable_tech = (
    gm_gdf[
        [
            "building_geometry",
            "1st_most_suitable_solution",
            "LAD23NM",
            "max_contiguous_outdoor_space_area_m2",
            "property_type_flat",
            "in_block_of_flats",
            "in_city_centre",
            # keeping the in_hn_zone column for pipeline changes afterwards
            "in_hn_zone",
            "UPRN",
        ]
    ]
    .groupby(["building_geometry", "1st_most_suitable_solution"])
    .agg(
        {
            "LAD23NM": "first",
            "property_type_flat": "sum",
            "UPRN": "count",
            "max_contiguous_outdoor_space_area_m2": "mean",
            "in_block_of_flats": "first",
            "in_city_centre": "first",
            "in_hn_zone": "first",
        }
    )
    .reset_index()
    .drop_duplicates(
        # we drop duplicates based on building geometry and most suitable solution, as we're assuming one solution per building
        ["building_geometry", "1st_most_suitable_solution"]
    )
    .rename(columns={"building_geometry": "geometry"})
)
gm_building_most_suitable_tech["max_contiguous_outdoor_space_area_m2"] = (
    gm_building_most_suitable_tech["max_contiguous_outdoor_space_area_m2"].round()
)
gm_building_most_suitable_tech["percent_flats"] = (
    gm_building_most_suitable_tech["property_type_flat"]
    / gm_building_most_suitable_tech["UPRN"]
) * 100

# Convert back to a GeoDataFrame
gm_building_most_suitable_tech = gpd.GeoDataFrame(
    gm_building_most_suitable_tech, geometry="geometry", crs=gm_gdf.crs
)

# %%
gm_building_most_suitable_tech

# %%
# These numbers should be the same, not sure why they are not
gm_building_most_suitable_tech["geometry"].nunique(), len(
    gm_building_most_suitable_tech
)

# %%
gm_building_most_suitable_tech["1st_most_suitable_solution"].value_counts(dropna=False)

# %%
gm_gdf[pd.isnull(gm_gdf["1st_most_suitable_solution"])]

# %%
map_building_techs_in_ward(
    techs_gdf=gm_building_most_suitable_tech,
    col="1st_most_suitable_solution",
    specific_ward_gdf=gm_la_boundaries_gdf,
    colours=colours,
    ward="Greater Manchester LAs",
    map_="Predicted",
)

# %%
# # Save most suitable tech per building locally as kml
# gm_building_most_suitable_tech[["geometry", "1st_most_suitable_solution"]].to_file("greater_manchester_las_building_most_suitable_tech.kml", driver="KML")

# %%
# There are 2 buildings without geometry... needs further investigation. For now we remove them!
gm_building_most_suitable_tech[pd.isnull(gm_building_most_suitable_tech["geometry"])]

# %%
# Removing buildings with no geometry
gm_building_most_suitable_tech = gm_building_most_suitable_tech[
    ~pd.isnull(gm_building_most_suitable_tech["geometry"])
]

# %%
# Adding colour column for visualisation purposes
gm_building_most_suitable_tech["color"] = gm_building_most_suitable_tech[
    "1st_most_suitable_solution"
].map(colours)

# %%
for lad in gm_building_most_suitable_tech["LAD23NM"].unique():
    print("Generating map for:", lad)
    lad_gdf = gm_building_most_suitable_tech[
        gm_building_most_suitable_tech["LAD23NM"] == lad
    ]
    # 1. Ensure the CRS is correct for Folium
    lad_gdf = lad_gdf.to_crs(epsg=4326)

    # 2. Initialize the map
    # Centering based on the average coordinates of the polygons
    map = folium.Map(
        location=[
            lad_gdf.geometry.centroid.y.mean(),
            lad_gdf.geometry.centroid.x.mean(),
        ],
        zoom_start=12,
    )

    # 3. Create the Popup object
    popup = folium.GeoJsonPopup(
        fields=[
            "property_type_flat",
            "UPRN",
            "percent_flats",
            "in_block_of_flats",
            "max_contiguous_outdoor_space_area_m2",
            "in_city_centre",
            "in_hn_zone",
            "1st_most_suitable_solution",
        ],
        aliases=[
            "Number of flats:",
            "Number of properties in building:",
            "Percent of flats (%):",
            "Property in block of flats:",
            "Avg max outdoor space area (m2):",
            "In city centre:",
            "In HN zone:",
            "Most suitable solution:",
        ],
        localize=True,
        labels=True,
        style="background-color: white;",
    )

    # 3. Add polygons using the color col
    folium.GeoJson(
        lad_gdf,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["color"],
            "color": feature["properties"][
                "color"
            ],  # Outline color (otherwise it is blue for all polygons)
            "fillOpacity": 1,  # Opacity
        },
        popup=popup,
    ).add_to(map)

    map.save(
        os.path.join(
            PROJECT_DIR,
            "outputs",
            f"greater_manchester_lad_{lad}_most_suitable_tech_per_building.html",
        )
    )

# %%
# Creating a version of the map where all properties within HNZ are labelled as district heat network, even where that's not the output of our decision tree
gdf_gm_default_hn = gm_building_most_suitable_tech.copy()
gdf_gm_default_hn["1st_most_suitable_solution"] = gdf_gm_default_hn.apply(
    lambda x: (
        "District heat network" if x["in_hn_zone"] else x["1st_most_suitable_solution"]
    ),
    axis=1,
)
gdf_gm_default_hn["color"] = gdf_gm_default_hn.apply(
    lambda x: "#EA2541" if x["in_hn_zone"] else x["color"], axis=1
)

# %%
for lad in gdf_gm_default_hn["LAD23NM"].unique():
    print("Generating map for:", lad)
    lad_gdf = gdf_gm_default_hn[gdf_gm_default_hn["LAD23NM"] == lad]

    # 1. Ensure the CRS is correct for Folium
    lad_gdf = lad_gdf.to_crs(epsg=4326)

    # 2. Initialize the map
    # Centering based on the average coordinates of the polygons
    map_gm_default_hn = folium.Map(
        location=[
            lad_gdf.geometry.centroid.y.mean(),
            lad_gdf.geometry.centroid.x.mean(),
        ],
        zoom_start=12,
    )

    # 3. Create the Popup object
    popup = folium.GeoJsonPopup(
        fields=[
            "property_type_flat",
            "UPRN",
            "percent_flats",
            "in_block_of_flats",
            "max_contiguous_outdoor_space_area_m2",
            "in_city_centre",
            "in_hn_zone",
            "1st_most_suitable_solution",
        ],
        aliases=[
            "Number of flats:",
            "Number of properties in building:",
            "Percent of flats (%):",
            "Property in block of flats:",
            "Avg max outdoor space area (m2):",
            "In city centre:",
            "In HN zone:",
            "Most suitable solution:",
        ],
        localize=True,
        labels=True,
        style="background-color: white;",
    )

    # 3. Add polygons using the color col
    folium.GeoJson(
        lad_gdf,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["color"],
            "color": feature["properties"]["color"],
            "fillOpacity": 1,  # Opacity
        },
        popup=popup,
    ).add_to(map_gm_default_hn)

    map_gm_default_hn.save(
        os.path.join(
            PROJECT_DIR,
            "outputs",
            f"greater_manchester_lad_{lad}_default_hnz_most_suitable_tech_per_building.html",
        )
    )

# %% [markdown]
# ## 6. Share of homes per technology
#
# These stats are up to date (after reassigning some solutions based on other properties in the same building footprint).

# %%
# Distribution of properties for the 1st most suitable solutions (in %)
gm_gdf["1st_most_suitable_solution"].value_counts(normalize=True) * 100

# %%
# Distribution of properties for the 1st most suitable solutions (in counts)
gm_gdf["1st_most_suitable_solution"].value_counts()

# %%
# Creating a version of the gdf where all properties within HNZ are labelled as district heat network, even where that's not the output of our decision tree
gdf_gm_default_hn = gm_gdf.copy()
gdf_gm_default_hn["1st_most_suitable_solution"] = gdf_gm_default_hn.apply(
    lambda x: (
        "District heat network" if x["in_hn_zone"] else x["1st_most_suitable_solution"]
    ),
    axis=1,
)

# %%
# Distribution of properties for the 1st most suitable solutions (in %) - with HN zones as district heat network
gdf_gm_default_hn["1st_most_suitable_solution"].value_counts(normalize=True) * 100

# %%
# Distribution of properties for the 1st most suitable solutions (in counts) - with HN zones as district heat network
gdf_gm_default_hn["1st_most_suitable_solution"].value_counts()

# %% [markdown]
# ## 7. Saving data

# %%
gm_building_most_suitable_tech.to_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/greater_manchester_las_building_most_suitable_tech.parquet"
)

# %%
# This is how data can be loaded, for future reference
# import geopandas as gpd
# f = gpd.read_parquet("s3://asf-heat-pump-suitability/local_heat_planning/outputs/greater_manchester_las_building_most_suitable_tech.parquet")

# %%
gm_building_most_suitable_tech

# %%
for lad in gm_building_most_suitable_tech["LAD23NM"].unique():
    lad_gdf = gm_building_most_suitable_tech[
        gm_building_most_suitable_tech["LAD23NM"] == lad
    ]
    lad_gdf.to_file(
        f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{lad.lower()}_building_most_suitable_tech.geojson",
        driver="GeoJSON",
    )

# %%
# Names of geojson files saved in the S3 bucket
import boto3

s3 = boto3.resource("s3")
bucket = s3.Bucket("asf-heat-pump-suitability")

objects = [obj for obj in bucket.objects.filter(Prefix="local_heat_planning/outputs/")]
object_keys = [obj.key for obj in objects if obj.key.endswith(".geojson")]

# %%
