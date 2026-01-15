# %% [markdown]
# # Visualising the results of the decision tree
#
# In this notebook we:
# - Code the decision tree that outputs the most suitable tech for each property given the following inputs: wether it is in a city centre or planned HN zone, garden size and whether the property is in a block of flats or not.
# - Map the outputs of the decision tree for all domestic buildings in Plymouth
# - Compare the results of the decision tree for one ward that was manually labelled for which tech is most suitable

# %%
# package imports
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium

# %%
# local imports
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
}

# %% [markdown]
# ## 1. Loading data

# %% [markdown]
# ### 1.1. Blocks of flats flag for each domestic property
#
# The dataset contains a flag for whether each domestic property is in a block of flats or not. Additionally, it contains info about:
# - wether the building where the property is located contains flats
# - the confidence associated with the blocks of flats label for each property
# - the building type (e.g. building with flats but not a block, blocks of flats, building without flats)
#
# This dataset contains data for Plymouth and other sampling areas.

# %%
blocks_of_flats = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/sampling_areas_residential_uprns_with_block_of_flats.parquet"
)
# Convert to pandas for compatibility with geopandas
blocks_of_flats = blocks_of_flats.to_pandas()

# %%
blocks_of_flats.head()

# %%
# Checking the average, median and minimum confidence of the block of flats label for blocks and non-blocks
blocks_of_flats.groupby("building_type")[["block_of_flats_label_proba"]].agg(
    ["mean", "median", "min"]
)

# %%
# Checking the distribution of building types
blocks_of_flats.groupby("building_type")[["block_of_flats_label_proba"]].size()

# %% [markdown]
# ### 1.2. Manually labelled data for one specific ward
#
# We have manually clustered domestic properties in one Plymouth ward and assigned the most suitable heating technology for each cluster.
# This dataset contains the geometries of the clusters and the most suitable heating technology for each cluster.

# %%
labelled_tech = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/plymouth_spcific_ward_labelled_technology_polygons.kml"
)

# %%
labelled_tech.head()

# %%
# Minor processing to change names of tech
labelled_tech["Name"] = labelled_tech["Name"].map(
    {
        "SGL": "Networked GSHP",
        "ASHP": "Individual solution",
        "Communal HN or SGL": "Communal solutions",
    }
)

# %%
labelled_tech.crs

# %%
# Update CRS to EPSG:27700
labelled_tech = labelled_tech.to_crs(epsg=27700)

# %% [markdown]
# ### 1.3. Plymouth residential UPRNs data
#
# Plymouth residential UPRNs data contains info about:
# - each residential UPRN in Plymouth
# - X and Y coordinates of each UPRN (as well as latitude and longitude)

# %%
# Getting data and converting polars df to geodf
plymouth_uprns = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns.parquet"
)
plymouth_uprns = generate_gdf_uprn_coords(df=plymouth_uprns)
plymouth_uprns.head()

# %% [markdown]
# ### 1.4. City centres and planned HNZ
#
# For each domestic UPRN `city_centre_data` contains:
# - X and Y coordinates of each UPRN
# - `in_city_centre`: a flag for whether it is located in a city centre according to a set of pre-defined spatial signature types (as per the Spatial Signatures Framework)
# - `spatial_signature_type`: the respective spatial signature type for each UPRN
#
# For each domestic UPRN `hnz_data` contains:
# - X and Y coordinates of each UPRN
# - `in_hn_zone`: a flag for whether it is located in a planned DESNZ heat network zone
#

# %%
# Getting data and converting polars df to geodf
city_centre_data = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_city_centres.parquet"
)
city_centre_data = generate_gdf_uprn_coords(df=city_centre_data)
city_centre_data.head()

# %%
# Getting data and converting polars df to geodf
hnz_data = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_hn_zones.parquet"
)
hnz_data = generate_gdf_uprn_coords(df=hnz_data)
hnz_data.head()

# %% [markdown]
# ### 1.5. Plymouth features dataset
#
# Plymouth features dataset will eventually contain all the features we need to run the decision tree for each domestic UPRN in Plymouth including garden size, blocks of flats flag, city centre flag and planned HNZ flag. For now, it only contains:
# - each residential UPRN in Plymouth
# - X and Y coordinates of each UPRN
# - wether property is a flat
# - NATIONALCADASTRALREFERENCE which identifies the specific building where the property is located
# - max contiguous and total outdoor space in m2

# %%
# Getting data and converting polars df to geodf
plymouth_with_features = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_features.parquet"
)
plymouth_with_features = generate_gdf_uprn_coords(df=plymouth_with_features)
plymouth_with_features.head()

# %% [markdown]
# ### 1.5. Building footprints
#
# Building footprints dataset contains the geometries of all buildings in grid square `SX` including residential and non-residential buildings. This includes an area bigger than Plymouth boundary, so it needs to be filtered to the Plymouth boundary.
#
# It includes building footprint IDs and geometries of all buildings in grid square `SX`. To note that one building footprint ID sometimes contains multiple buildings.
#
# These are updated regularly by Ordnance Survey and with each update the building footprint IDs might change.

# %%
# Loading building footprints for grid square "SX"
building_footprints = load_gdf_os_openmap_local_layer(
    layer="building", grid_squares="SX"
)
building_footprints.head()

# %% [markdown]
# ### 1.6. Specific ward geometry
#
# This geojson contains the geometry of the specific ward in Plymouth where we have manually labelled the most suitable heating technology for clusters of domestic properties.

# %%
specific_ward_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/source_data/plymouth_specific_ward_boundary.geojson"
)

# %%
specific_ward_gdf.crs

# %%
# Update CRS to EPSG:27700
specific_ward_gdf.to_crs(epsg=27700, inplace=True)

# %%
specific_ward_gdf.head()

# %%
ward = specific_ward_gdf.name.values[0]

# %% [markdown]
# ### 1.7. Plymouth boundary

# %%
# Loading the Plymouth LA boundary
plymouth_la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
    select_las="plymouth"
)

# %%
plymouth_la_boundaries_gdf.head()

# %% [markdown]
# ### 1.7. Planned HN zones geometries
#
# This dataset contains the geometries of all planned heat network zones in Plymouth as per DESNZ data.

# %%
plymouth_hn_zones_gdf = load_gdf_heat_network_zones(local_authority="plymouth")

# %%
plymouth_hn_zones_gdf.crs

# %%
plymouth_hn_zones_gdf.head()

# %% [markdown]
# ## 2. Processing data
#
# Most of the processing steps below are temporary as there will be pipelines that will put all of this data together into one Plymouth features dataset.

# %% [markdown]
# ### 2.1. [temporary] Joining all dfs into one geodf

# %%
plymouth_uprns.set_index("UPRN", inplace=True)
plymouth_with_features.set_index("UPRN", inplace=True)
city_centre_data.set_index("UPRN", inplace=True)
hnz_data.set_index("UPRN", inplace=True)
blocks_of_flats.set_index("UPRN", inplace=True)

# %%
plymouth_uprns.head()

# %%
# checking if lengths are the same
len(plymouth_uprns), len(plymouth_with_features), len(city_centre_data), len(hnz_data)

# %%
# Joining all geodfs into one based on UPRN
plymouth_gdf = plymouth_uprns.join(city_centre_data[["in_city_centre"]])
plymouth_gdf = plymouth_gdf.join(hnz_data[["in_hn_zone"]])
plymouth_gdf = plymouth_gdf.join(
    plymouth_with_features.drop(columns=["X_COORDINATE", "Y_COORDINATE", "geometry"])
)
plymouth_gdf = plymouth_gdf.join(
    blocks_of_flats.rename(columns={"block_of_flats": "in_block_of_flats"})[
        ["in_block_of_flats", "block_of_flats_label_proba", "building_type"]
    ],
    how="left",
)

# %%
# for properties without flats, fill NaN with 1 to represent 100% confidence they are not flats
plymouth_gdf["block_of_flats_label_proba"].fillna(1, inplace=True)

# %%
plymouth_gdf.reset_index(inplace=True)

# %%
plymouth_gdf.head()

# %% [markdown]
# ### 2.2. [temporary] Identifying properties in blocks of flats
#
# Code has now been commented out - this was prior to having the blocks of flats flag data. Doesn't need to be reviewed.

# %%
# counts of properties per building
# counts = plymouth_df.groupby("NATIONALCADASTRALREFERENCE").count()


## Roisin's suggestion (I ended up doing something easier, based on UPRN counts per land parcel)
# groupby NATIONALCADASTRALREFERENCE (land parcel) - if garden size > Xm2 and Y number of UPRNs then block (for now)

# NATIONALCADASTRALREFERENCE: ID of the land parcel
# A land extent should represent a section of a building, rather than the whole building footprint
# every uprn that shares a garden is in the same land extent

# %%
# While we wait for the modelled data, we identify blocks of flats as those with 6 or more UPRNs per land parcel
# blocks_of_flats_ncref = counts[counts["UPRN"]>=6].index.tolist()

# plymouth_df["is_in_block_of_flats"] = plymouth_df["NATIONALCADASTRALREFERENCE"].apply(lambda x: x in blocks_of_flats_ncref)

# %% [markdown]
# ### 2.3. Spatial join to add building footprints to main geodf
#
# Adding geometries of building footprints in Plymouth to the main geodf.

# %%
# we create a copy of the (building) geometry column so that we can keep it after the spatial join
plymouth_gdf["property_geometry"] = plymouth_gdf.geometry
building_footprints["building_geometry"] = building_footprints["geometry"]
plymouth_gdf = plymouth_gdf.sjoin(
    building_footprints[["geometry", "building_geometry"]],
    how="left",
    predicate="within",
).drop(columns=["index_right"])

# %% [markdown]
# ### 2.4. Spatial join with specific ward
#
# Adding a flag column to main geodf to identify properties located in the specific ward.

# %%
specific_ward_gdf["ward"] = ward
plymouth_gdf = plymouth_gdf.sjoin(
    specific_ward_gdf[["ward", "geometry"]], how="left", predicate="within"
).drop(columns=["index_right"])

# %% [markdown]
# ### 2.5. Spatial join with DESNZ HN zones geometries data
#
# For each property in a planned HN zone, we add the HN zone geometry.

# %%
# Adding a column to indicate if the property is within a DESNZ heat network zone
plymouth_hn_zones_gdf["desnz_hn_zone"] = True

# Creating a copy of the geometry column to keep after the spatial join
plymouth_hn_zones_gdf["desnz_hn_zone_geometry"] = plymouth_hn_zones_gdf["geometry"]

# %%
plymouth_gdf = plymouth_gdf.sjoin(
    plymouth_hn_zones_gdf[["geometry", "desnz_hn_zone_geometry"]],
    how="left",
    predicate="within",
)

# %% [markdown]
# ### 2.6. Replacing any labelled data in HN zones (as one of the other solutions) as district HN
# Some areas that are covered by HN zones have been manually labelled as not HN, so we need to replace those labels.

# %%
labelled_tech = labelled_tech.sjoin(
    plymouth_hn_zones_gdf[["desnz_hn_zone", "geometry", "desnz_hn_zone_geometry"]],
    how="left",
    predicate="intersects",
).drop(columns=["index_right"])

labelled_tech["Name"] = np.where(
    labelled_tech["desnz_hn_zone"] == True,
    "District heat network",
    labelled_tech["Name"],
)

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
            if garden_size > 70:
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
            if garden_size > 30:
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
plymouth_gdf["in_city_centre_or_hn_zone"] = (
    plymouth_gdf["in_city_centre"] | plymouth_gdf["in_hn_zone"]
)

plymouth_gdf["most_suitable_solutions"] = plymouth_gdf.apply(
    lambda x: define_decision_tree(
        x["in_block_of_flats"],
        x["max_contiguous_outdoor_space_area_m2"],
        x["in_city_centre_or_hn_zone"],
    ),
    axis=1,
)

# %%
plymouth_gdf["1st_most_suitable_solution"] = plymouth_gdf[
    "most_suitable_solutions"
].apply(lambda x: x[1])
plymouth_gdf["2nd_most_suitable_solution"] = plymouth_gdf[
    "most_suitable_solutions"
].apply(lambda x: x[2])
plymouth_gdf["decision_tree_path"] = plymouth_gdf["most_suitable_solutions"].apply(
    lambda x: x["path"]
)

# %% [markdown]
# ## 4. Distribution of garden sizes for each path of the decision tree
#
# Gardens sizes above 500m2 are removed from the analysis as we're mostly focused on seeing the distribution of garden sizes around 30 and 70m2.

# %%
len(plymouth_gdf[plymouth_gdf["max_contiguous_outdoor_space_area_m2"] > 500]) / len(
    plymouth_gdf
)

# %%
# df of garden sizes for different decision tree paths (for gardens <= 500m2)
garden_sizes_decision_tree_path = pd.DataFrame(
    plymouth_gdf[plymouth_gdf["max_contiguous_outdoor_space_area_m2"] <= 500]
    .groupby(["in_block_of_flats", "in_city_centre_or_hn_zone"])[
        "max_contiguous_outdoor_space_area_m2"
    ]
    .apply(list)
)

# %%
garden_sizes_decision_tree_path.reset_index(inplace=True)
garden_sizes_decision_tree_path

# %%
for path in garden_sizes_decision_tree_path.index:
    plt.figure(figsize=(8, 2))
    plt.hist(
        garden_sizes_decision_tree_path.loc[
            path, "max_contiguous_outdoor_space_area_m2"
        ],
        bins=range(0, 500, 10),
        color="skyblue",
        edgecolor="black",
        density=True,
    )
    plt.title(
        f"Block: {garden_sizes_decision_tree_path.loc[path, 'in_block_of_flats']}, HNZ or city centre: {garden_sizes_decision_tree_path.loc[path, 'in_city_centre_or_hn_zone']}"
    )
    plt.xlabel("Max contiguous outdoor space area (m²)")
    # vertical line at 30 and 70
    plt.axvline(x=30, color="red", linestyle="--", label="30 m² threshold")
    plt.axvline(x=70, color="green", linestyle="--", label="70 m² threshold")

# %%


# %% [markdown]
# ## 5. Number of solutions per land parcel and building footprint
#
# In this section we check wether there are multiple solutions for properties located in the same land parcel and building footprint. Ideally, all properties in the same land parcel & footprint should have the same solution.
#
# As we can observe below, most land parcels and building footprints have one solution for all properties, but there are some buildings with multiple solutions.

# %%
# Distribution of unique 1st most suitable solutions per land parcel (in counts)
plymouth_gdf.groupby("NATIONALCADASTRALREFERENCE")[
    ["1st_most_suitable_solution"]
].nunique()["1st_most_suitable_solution"].value_counts()

# %%
# Distribution of unique 1st most suitable solutions per land parcel (in proportions)
plymouth_gdf.groupby("NATIONALCADASTRALREFERENCE")[
    ["1st_most_suitable_solution"]
].nunique()["1st_most_suitable_solution"].value_counts(normalize=True)

# %%
# Checking distribution for properties within the specific ward
plymouth_gdf[~pd.isnull(plymouth_gdf["ward"])].groupby("NATIONALCADASTRALREFERENCE")[
    ["1st_most_suitable_solution"]
].nunique()["1st_most_suitable_solution"].value_counts()

# %%
# Number of unique land parcels in Plymouth vs. number of buildings
# This shows that lots of land parcels are aggregated into one single building geometry/ footprint
plymouth_gdf["NATIONALCADASTRALREFERENCE"].nunique(), plymouth_gdf[
    "building_geometry"
].nunique()

# %%
# Identifying pairs of different 1st most suitable solutions per land parcel
solutions_per_land_parcel = plymouth_gdf.groupby("NATIONALCADASTRALREFERENCE")[
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
solutions_per_footprint = plymouth_gdf.groupby("building_geometry")[
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
land_parcel_to_footprint = plymouth_gdf.groupby("NATIONALCADASTRALREFERENCE")[
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
# This is currently based on Plymouth only (both pairs found in land parcels and building footprints)
# but needs to be generalised to all possible cases


def assign_unique_sol(solution_set: set) -> str:
    """
    Assigns a unique solution based on the combination of solutions in the set.

    Args:
        solution_set (set): A set of solutions.

    Returns:
        str: A unique solution assigned based on the combination.
    """
    if solution_set == {"District heat network", "Individual solution"}:
        return "District heat network"
    elif solution_set == {"Networked GSHP", "Individual solution"}:
        return "Networked GSHP"
    elif solution_set == {
        "District heat network",
        "Individual solution",
        "Communal solutions",
    }:
        return "District heat network"
    elif solution_set == {
        "District heat network",
        "Individual solution",
        "Networked GSHP",
    }:
        return "District heat network"
    elif solution_set == {"Individual solution", "Communal solutions"}:
        return "Communal solutions"
    elif solution_set == {"Networked GSHP", "Communal solutions"}:
        return "Networked GSHP"


solutions_per_footprint["final_solution"] = solutions_per_footprint[
    "1st_most_suitable_solution"
].apply(lambda x: assign_unique_sol(x))

# %%
mapping_set_to_final_solution = solutions_per_footprint.set_index("building_geometry")[
    "final_solution"
].to_dict()

# %%
# Mapping properties in the same land parcel with different 1st most suitable solutions (for one ward)
fig, ax = plt.subplots(figsize=(15, 8))

plot_mult_solutions_ = plymouth_gdf[
    plymouth_gdf["NATIONALCADASTRALREFERENCE"].isin(
        solutions_per_land_parcel["NATIONALCADASTRALREFERENCE"]
    )
]
plot_mult_solutions_ = plot_mult_solutions_[~pd.isna(plot_mult_solutions_["ward"])]

build_geom = (
    plot_mult_solutions_[["building_geometry"]]
    .rename(columns={"building_geometry": "geometry"})
    .drop_duplicates()
)
build_geom.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)
# mapping individual properties with the colour of the most suitable solution
plot_mult_solutions_.plot(
    ax=ax,
    column="1st_most_suitable_solution",
    categorical=True,
    legend=True,
    markersize=1,
    color=plot_mult_solutions_["1st_most_suitable_solution"].map(colours),
)

# mapping ward boundary
specific_ward_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)


# title and legend
ax.set_title(
    f"Properties in the same land parcel with different techs assigned", fontsize=12
)
handles = [
    mpatches.Patch(facecolor=colours[tech], label=tech) for tech in colours.keys()
]
ax.legend(handles=handles, loc="upper right")

# %%
# Mapping properties in the same building footprint with different 1st most suitable solutions (for one ward)
fig, ax = plt.subplots(figsize=(15, 8))

plot_mult_solutions_ = plymouth_gdf[
    plymouth_gdf["building_geometry"].isin(solutions_per_footprint["building_geometry"])
]
plot_mult_solutions_ = plot_mult_solutions_[~pd.isna(plot_mult_solutions_["ward"])]

build_geom = (
    plot_mult_solutions_[["building_geometry"]]
    .rename(columns={"building_geometry": "geometry"})
    .drop_duplicates()
)
build_geom.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)
# mapping individual properties with the colour of the most suitable solution
plot_mult_solutions_.plot(
    ax=ax,
    column="1st_most_suitable_solution",
    categorical=True,
    legend=True,
    markersize=1,
    color=plot_mult_solutions_["1st_most_suitable_solution"].map(colours),
)

# mapping ward boundary
specific_ward_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)


# title and legend
ax.set_title(
    f"Properties in the same building with different techs assigned", fontsize=12
)
handles = [
    mpatches.Patch(facecolor=colours[tech], label=tech) for tech in colours.keys()
]
ax.legend(handles=handles, loc="upper right")

# %%
plymouth_gdf["1st_most_suitable_solution"] = plymouth_gdf.apply(
    lambda x: (
        mapping_set_to_final_solution[x["building_geometry"]]
        if x["building_geometry"] in mapping_set_to_final_solution
        else x["1st_most_suitable_solution"]
    ),
    axis=1,
)

# %% [markdown]
# ## 6. Visualising "predictions" against labelled data in one ward that was manually labelled

# %%
ward_gdf = plymouth_gdf[plymouth_gdf["ward"] == ward]


# %%
def map_suitable_tech_vs_labelled_tech(
    tech: str,
    ward_df: gpd.GeoDataFrame,
    specific_ward_gdf: gpd.GeoDataFrame,
    labelled_tech: gpd.GeoDataFrame,
    colours: dict,
):
    """
    Maps properties in a specific ward that are most suitable for a given technology, overlaying labelled data polygons.

    Args:
        tech (str): a low carbon heating solution in the set {"Individual solution", "Networked GSHP", "Communal solutions", "District heat network"}
        ward_df (gpd.GeoDataFrame): ward data with most suitable solutions
        specific_ward_gdf (gpd.GeoDataFrame): specific ward boundary data
        labelled_tech (gpd.GeoDataFrame): labelled technology polygons data
        colours (dict): mapping of technologies to colours
    """
    tech_specific_df = ward_df[ward_df["1st_most_suitable_solution"] == tech]
    fig, ax = plt.subplots(figsize=(15, 8))

    # mapping individual properties with the colour of the most suitable solution
    tech_specific_df.plot(
        ax=ax,
        column="1st_most_suitable_solution",
        categorical=True,
        legend=True,
        color=tech_specific_df["1st_most_suitable_solution"].map(colours),
        markersize=2,
    )

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

    # mapping ward boundary
    specific_ward_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

    # title and legend
    ax.set_title(f"Properties in {ward} most suitable for {tech.lower()}", fontsize=12)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Properties",
            markerfacecolor=colours[tech],
            markersize=5,
        ),
        mpatches.Patch(
            facecolor=colours[tech],
            edgecolor="black",
            alpha=0.5,
            label="Labelled data",
            linestyle="--",
        ),
    ]
    ax.legend(handles=handles, loc="upper right")


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


# %% [markdown]
# ### 6.1. Individual solution

# %%
# The idea behind this analysis was to see if lower confidence in the blocks of flats label correlated with disagreement
# between assigned most suitable solution and labelled data, which doesn't seem to be the case
map_blocks_of_flats_prob(
    tech="Individual solution",
    ward_df=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
# Mapping properties most suitable for individual solutions vs labelled data
map_suitable_tech_vs_labelled_tech(
    tech="Individual solution",
    ward_df=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %% [markdown]
# ### 6.2. Networked GSHP

# %%
map_blocks_of_flats_prob(
    tech="Networked GSHP",
    ward_df=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
map_suitable_tech_vs_labelled_tech(
    tech="Networked GSHP",
    ward_df=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %% [markdown]
# ### 6.3. Communal solutions

# %%
map_blocks_of_flats_prob(
    tech="Communal solutions",
    ward_df=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
map_suitable_tech_vs_labelled_tech(
    tech="Communal solutions",
    ward_df=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %% [markdown]
# ### 6.4. District heat network

# %%
ward_hnz_join = plymouth_hn_zones_gdf.sjoin(
    specific_ward_gdf, how="inner", predicate="intersects"
)
intersection_shape = gpd.overlay(ward_hnz_join, specific_ward_gdf, how="intersection")


map_blocks_of_flats_prob(
    tech="District heat network",
    ward_df=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=intersection_shape,
    colours=colours,
)

map_suitable_tech_vs_labelled_tech(
    tech="District heat network",
    ward_df=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=intersection_shape,
    colours=colours,
)

# %%
# Properties with confidence lower than 0.6 of being a block of flats across all technologies in the whole of Plymouth
map_blocks_of_flats_prob(
    tech="",
    ward_df=plymouth_gdf,
    specific_ward_gdf=plymouth_la_boundaries_gdf,
    threshold=0.6,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %% [markdown]
# ## 7. Assessing the results of the decision tree for the labelled data in one ward
#
# We shouldn't expect a perfect match between the decision tree outputs and the manually labelled data.

# %%
# For each UPRN, we have its geometry, the most suitable solution and the label
labelled_ward_stats = (
    ward_gdf[["UPRN", "geometry", "1st_most_suitable_solution"]]
    .sjoin(labelled_tech[["Name", "geometry"]], how="left", predicate="within")
    .drop(columns=["index_right"])
    .rename(
        columns={
            "Name": "labelled_tech",
            "1st_most_suitable_solution": "most_suitable_solution",
        }
    )
)

# %%
labelled_ward_stats

# %%
confusion_matrix = pd.crosstab(
    labelled_ward_stats["labelled_tech"],
    labelled_ward_stats["most_suitable_solution"],
    rownames=["Labelled"],
    colnames=["Predicted"],
    dropna=True,
)
confusion_matrix

# %%
recall = {}
precision = {}

for tech in confusion_matrix.index:
    true_positives = confusion_matrix.loc[tech, tech]
    false_negatives = confusion_matrix.loc[tech].sum() - true_positives
    false_positives = confusion_matrix[tech].sum() - true_positives
    recall[tech] = true_positives / (true_positives + false_negatives)
    precision[tech] = true_positives / (true_positives + false_positives)

# %%
recall

# %%
precision

# %% [markdown]
# ## 8. Visualising results per building in the labelled ward

# %%
# Estimated most suitable
ward_building_most_suitable_tech = (
    ward_gdf[
        [
            "building_geometry",
            "1st_most_suitable_solution",
            "2nd_most_suitable_solution",
            "decision_tree_path",
        ]
    ]
    .drop_duplicates()
    .rename(columns={"building_geometry": "geometry"})
)
ward_building_most_suitable_tech.head()


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
map_building_techs_in_ward(
    techs_gdf=ward_building_most_suitable_tech,
    col="1st_most_suitable_solution",
    specific_ward_gdf=specific_ward_gdf,
    colours=colours,
    ward=ward,
    map_="Predicted",
)

# %%
# Labelled most suitable
ward_building_labelled_tech = (
    ward_gdf[["UPRN", "building_geometry", "geometry"]]
    .sjoin(labelled_tech[["Name", "geometry"]], how="left", predicate="within")
    .drop(columns=["geometry", "index_right"])
    .drop_duplicates()
    .rename(columns={"building_geometry": "geometry", "Name": "labelled_tech"})
)
ward_building_labelled_tech.head()

# %%
# some buildings don't have labelled tech, because in a previous iteration of the HN zones, this area was part of the planned HN
fig, ax = plt.subplots(figsize=(15, 8))
ward_building_labelled_tech[ward_building_labelled_tech["labelled_tech"].isnull()].plot(
    ax=ax
)
specific_ward_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

# %%
ward_building_labelled_tech.dropna(inplace=True)

# %%
map_building_techs_in_ward(
    techs_gdf=ward_building_labelled_tech,
    col="labelled_tech",
    specific_ward_gdf=specific_ward_gdf,
    colours=colours,
    ward=ward,
    map_="Labelled",
)

# %%


# %% [markdown]
# ## 8. Visualising results per building in Plymouth

# %%
# Estimated most suitable
plymouth_building_most_suitable_tech = (
    plymouth_gdf[
        [
            "building_geometry",
            "1st_most_suitable_solution",
            # keeping the in_hn_zone column for pipeline changes afterwards
            "in_hn_zone",
        ]
    ]
    .drop_duplicates(
        # we drop duplicates based on building geometry and most suitable solution, as we're assuming one solution per building
        ["building_geometry", "1st_most_suitable_solution"]
    )
    .rename(columns={"building_geometry": "geometry"})
)


# %%
plymouth_building_most_suitable_tech

# %%
# These numbers should be the same, not sure why they are not
plymouth_building_most_suitable_tech["geometry"].nunique(), len(
    plymouth_building_most_suitable_tech
)

# %%
map_building_techs_in_ward(
    techs_gdf=plymouth_building_most_suitable_tech,
    col="1st_most_suitable_solution",
    specific_ward_gdf=plymouth_la_boundaries_gdf,
    colours=colours,
    ward="Plymouth",
    map_="Predicted",
)

# %%
# Save most suitable tech per building locally as kml
plymouth_building_most_suitable_tech[
    ["geometry", "1st_most_suitable_solution"]
].to_file("plymouth_building_most_suitable_tech.kml", driver="KML")

# %%
# There are 2 buildings without geometry... needs further investigation. For now we remove them!
plymouth_building_most_suitable_tech[
    pd.isnull(plymouth_building_most_suitable_tech["geometry"])
]

# %%
# Removing buildings with no geometry
plymouth_building_most_suitable_tech = plymouth_building_most_suitable_tech[
    ~pd.isnull(plymouth_building_most_suitable_tech["geometry"])
]

# %%
# Adding colour column for visualisation purposes
plymouth_building_most_suitable_tech["color"] = plymouth_building_most_suitable_tech[
    "1st_most_suitable_solution"
].map(colours)

# %%
# 1. Ensure the CRS is correct for Folium
plymouth_building_most_suitable_tech = plymouth_building_most_suitable_tech.to_crs(
    epsg=4326
)

# 2. Initialize the map
# Centering based on the average coordinates of the polygons
map = folium.Map(
    location=[
        plymouth_building_most_suitable_tech.geometry.centroid.y.mean(),
        plymouth_building_most_suitable_tech.geometry.centroid.x.mean(),
    ],
    zoom_start=12,
)

# 3. Add polygons using the color col
folium.GeoJson(
    plymouth_building_most_suitable_tech,
    style_function=lambda feature: {
        "fillColor": feature["properties"]["color"],
        "color": feature["properties"][
            "color"
        ],  # Outline color (otherwise it is blue for all polygons)
        "fillOpacity": 1,  # Opacity
    },
).add_to(map)

map.save("plymouth_most_suitable_tech_per_building.html")

# %%
# Creating a version of the map where all properties within HNZ are labelled as district heat network, even where that's not the output of our decision tree
gdf_plymouth_cc = plymouth_building_most_suitable_tech.copy()
gdf_plymouth_cc["1st_most_suitable_solution"] = gdf_plymouth_cc.apply(
    lambda x: (
        "District heat network" if x["in_hn_zone"] else x["1st_most_suitable_solution"]
    ),
    axis=1,
)
gdf_plymouth_cc["color"] = gdf_plymouth_cc.apply(
    lambda x: "#EA2541" if x["in_hn_zone"] else x["color"], axis=1
)

# %%
# 1. Ensure the CRS is correct for Folium
gdf_plymouth_cc = gdf_plymouth_cc.to_crs(epsg=4326)

# 2. Initialize the map
# Centering based on the average coordinates of the polygons
map_plymouth_cc = folium.Map(
    location=[
        gdf_plymouth_cc.geometry.centroid.y.mean(),
        gdf_plymouth_cc.geometry.centroid.x.mean(),
    ],
    zoom_start=12,
)

# 3. Add polygons using the color col
folium.GeoJson(
    gdf_plymouth_cc,
    style_function=lambda feature: {
        "fillColor": feature["properties"]["color"],
        "color": feature["properties"]["color"],
        "fillOpacity": 1,  # Opacity
    },
).add_to(map_plymouth_cc)

map_plymouth_cc.save("plymouthcc_most_suitable_tech_per_building.html")

# %% [markdown]
# ## 9. Share of homes per technology
#
# These stats are up to date (after reassigning some solutions based on other properties in the same building footprint).

# %%
# Distribution of properties for the 1st most suitable solutions (in %)
plymouth_gdf["1st_most_suitable_solution"].value_counts(normalize=True) * 100

# %%
# Distribution of properties for the 1st most suitable solutions  (in counts)
plymouth_gdf["1st_most_suitable_solution"].value_counts()

# %%
# Creating a version of the gdf where all properties within HNZ are labelled as district heat network, even where that's not the output of our decision tree
plymouth_gdf_cc = plymouth_gdf.copy()
plymouth_gdf_cc["1st_most_suitable_solution"] = plymouth_gdf_cc.apply(
    lambda x: (
        "District heat network" if x["in_hn_zone"] else x["1st_most_suitable_solution"]
    ),
    axis=1,
)

# %%
# Distribution of properties for the 1st most suitable solutions (in %) - with HN zones as district heat network
plymouth_gdf_cc["1st_most_suitable_solution"].value_counts(normalize=True) * 100

# %%
# Distribution of properties for the 1st most suitable solutions (in counts) - with HN zones as district heat network
gdf_plymouth_cc["1st_most_suitable_solution"].value_counts()

# %% [markdown]
# ## 10. Saving data

# %%
plymouth_building_most_suitable_tech.to_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_building_most_suitable_tech.parquet"
)

# %%
# This is how data can be loaded, for future reference
# import geopandas as gpd
# f = gpd.read_parquet("s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_building_most_suitable_tech.parquet")

# %%
