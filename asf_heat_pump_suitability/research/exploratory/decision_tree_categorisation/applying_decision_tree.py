# %%


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
import matplotlib.colors as mcolors
import folium
import os

# local imports
from asf_heat_pump_suitability import PROJECT_DIR
from asf_heat_pump_suitability import config

# %%
# local imports
from asf_heat_pump_suitability.pipeline.transform.uprns import generate_gdf_uprn_coords

from asf_heat_pump_suitability.getters.load_geodata import (
    load_gdf_heat_network_zones,
    load_gdf_os_openmap_layer,
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
# ### 1.1. Blocks of flats flag for each domestic property
#
# The dataset contains a flag for whether each domestic property is in a block of flats or not. Additionally, it contains info about:
# - wether the building where the property is located contains flats
# - the confidence associated with the blocks of flats label for each property
# - the building type (e.g. building with flats but not a block, blocks of flats, building without flats)
#
# This dataset contains data for Plymouth and other sampling areas.

# %%
blocks_of_flats = pd.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/sampling_areas_residential_uprns_with_block_of_flats_label.parquet"
)

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
    config["output"]["dataset"]["domestic_uprns"].format(local_authority="plymouth")
)
plymouth_uprns = generate_gdf_uprn_coords(df=plymouth_uprns)
plymouth_uprns.head()

# %% [markdown]
# ### 1.4. City centres and planned HNZ
#
# For each domestic UPRN `hnz_and_city_centre_data` contains:
# - X and Y coordinates of each UPRN
# - `in_hn_zone`: a flag for whether it is located in a planned DESNZ heat network zone
# - `in_city_centre`: a flag for whether it is located in a city centre according to a set of pre-defined spatial signature types (as per the Spatial Signatures Framework)
# - `spatial_signature_type`: the respective spatial signature type for each UPRN

# %%
# Getting data and converting polars df to geodf
hnz_and_city_centre_data = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_hn_zones_city_centres.parquet"
)
hnz_and_city_centre_data = generate_gdf_uprn_coords(df=hnz_and_city_centre_data)
hnz_and_city_centre_data.head()

# %%


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
    config["output"]["dataset"]["domestic_uprns_with_features"].format(
        local_authority="plymouth"
    )
)
plymouth_with_features = generate_gdf_uprn_coords(df=plymouth_with_features)
plymouth_with_features.head()

# %% [markdown]
# ### 1.5. Building footprints
#
# Building footprints dataset contains the geometries of all buildings in grid square `SX` including residential and non-residential buildings. This includes an area bigger than Plymouth boundary, so it needs to be filtered to the Plymouth boundary.
#
# It includes building footprint IDs and geometries of all buildings in grid square `SX`. To note that one building footprint ID sometimes merges multiple buildings.
#
# These are updated regularly by Ordnance Survey and with each update the building footprint IDs changes.

# %%
# Loading building footprints for grid square "SX"
building_footprints = load_gdf_os_openmap_layer(layer="building", grid_squares="SX")
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
plymouth_hn_zones_gdf = load_gdf_heat_network_zones(boundary=plymouth_la_boundaries_gdf)

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
hnz_and_city_centre_data.set_index("UPRN", inplace=True)
blocks_of_flats.set_index("UPRN", inplace=True)

# %%
plymouth_uprns.head()

# %%
# checking if lengths are the same
len(plymouth_uprns), len(plymouth_with_features), len(hnz_and_city_centre_data)

# %%
# Joining all geodfs into one based on UPRN
plymouth_gdf = (
    plymouth_uprns.join(hnz_and_city_centre_data[["in_city_centre", "in_hn_zone"]])
    .join(
        plymouth_with_features.drop(
            columns=["X_COORDINATE", "Y_COORDINATE", "geometry"]
        )
    )
    .join(
        blocks_of_flats[["building_type"]],
        how="left",
    )
)

# %%
plymouth_gdf

# %%
plymouth_gdf.reset_index(inplace=True)

# %%
plymouth_gdf.head()

# %%


# %% [markdown]
# ### 2.3. Spatial join to add building footprints to main geodf
#
# Adding geometries of building footprints in Plymouth to the main geodf.

# %%
# we create a copy of the (building) geometry column so that we can keep it after the spatial join
plymouth_gdf["uprn_geometry"] = plymouth_gdf.geometry
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
def identify_dict_most_suitable_tech(
    in_block_of_flats: bool, outdoor_space: float, city_centre_or_hnz: bool
) -> dict:
    """
    Defines the decision tree to identify:
    - first and second most suitable low carbon heating solutions for each UPRN.
    - the path taken in the decision tree.

    Args:
        in_block_of_flats (bool): Whether the property is in a block of flats.
        outdoor_space (float): Outdoor space in square meters.
        city_centre_or_hnz (bool): Whether the property is in the city centre or in a planned heat network zone.

    Returns:
        dict: A dictionary with the first and second most suitable heating solutions and the path taken in the decision tree.
    """

    if in_block_of_flats:
        if city_centre_or_hnz:
            return {
                1: "District heat network",
                2: "Communal solutions",
                "path": "1. blocks of flats and HNZ/ city centre",
            }
        else:
            return {
                1: "Communal solutions",
                2: "Networked GSHP",
                "path": "2. blocks of flats, not  HNZ/ city centre",
            }
    else:
        if city_centre_or_hnz:
            if pd.isnull(outdoor_space):
                return {
                    1: "Individual solution or District heat network",
                    2: "Individual solution or District heat network",
                    "path": "Unknown outdoor space in city centre",
                }
            elif outdoor_space > 70:
                return {
                    1: "Individual solution",
                    2: "District heat network",
                    "path": "3. not blocks of flats, city centre, large outdoor space (70m2)",
                }
            else:
                return {
                    1: "District heat network",
                    2: "Networked GSHP",
                    "path": "4. not blocks of flats, city centre, small or no garden",
                }
        else:
            if pd.isnull(outdoor_space):
                return {
                    1: "Individual solution or Networked GSHP",
                    2: "Networked GSHP or Communal solutions",
                    "path": "Unknown outdoor space not in city centre",
                }
            elif outdoor_space > 30:
                return {
                    1: "Individual solution",
                    2: "Networked GSHP",
                    "path": "5. not blocks of flats, not city centre, large outdoor space (30m2)",
                }
            else:
                return {
                    1: "Networked GSHP",
                    2: "Communal solutions",
                    "path": "6. not blocks of flats, not city centre, small/no outdoor space",
                }


# %%
plymouth_gdf

# %%
plymouth_gdf["in_city_centre_or_hn_zone"] = (
    plymouth_gdf["in_city_centre"] | plymouth_gdf["in_hn_zone"]
)

plymouth_gdf["most_suitable_solutions"] = plymouth_gdf.apply(
    lambda x: identify_dict_most_suitable_tech(
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
plymouth_gdf["1st_most_suitable_solution"].unique()

# %%
plymouth_gdf[pd.isnull(plymouth_gdf["max_contiguous_outdoor_space_area_m2"])]

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
# ## 5. Number of solutions per building footprint
#
# In this section we check wether there are multiple solutions for properties located in the same building footprint. Ideally, all properties in the same building footprint should have the same solution.
#
# As we can observe below, most building footprints have one solution for all properties, but there are some buildings with multiple solutions.

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


# %% [markdown]
# In most cases:
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
        return None  # this shouldn't happen as all combinations should be covered by the above


solutions_per_footprint["assigned_tech"] = solutions_per_footprint[
    "1st_most_suitable_solution"
].apply(lambda x: assign_unique_sol(x))

# %%
mapping_set_to_assigned_tech = solutions_per_footprint.set_index("building_geometry")[
    "assigned_tech"
].to_dict()

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
        mapping_set_to_assigned_tech[x["building_geometry"]]
        if x["building_geometry"] in mapping_set_to_assigned_tech
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
    ward_gdf: gpd.GeoDataFrame,
    specific_ward_gdf: gpd.GeoDataFrame,
    labelled_tech: gpd.GeoDataFrame,
    colours: dict,
):
    """
    Maps properties in a specific ward that are most suitable for a given technology, overlaying labelled data polygons.

    Args:
        tech (str): a low carbon heating solution in the set {"Individual solution", "Networked GSHP", "Communal solutions", "District heat network"}
        ward_gdf (gpd.GeoDataFrame): ward data with most suitable solutions
        specific_ward_gdf (gpd.GeoDataFrame): specific ward boundary data
        labelled_tech (gpd.GeoDataFrame): labelled technology polygons data
        colours (dict): mapping of technologies to colours
    """
    tech_specific_gdf = ward_gdf[ward_gdf["1st_most_suitable_solution"] == tech]
    fig, ax = plt.subplots(figsize=(15, 8))

    # mapping individual properties with the colour of the most suitable solution
    tech_specific_gdf.plot(
        ax=ax,
        column="1st_most_suitable_solution",
        categorical=True,
        legend=True,
        color=tech_specific_gdf["1st_most_suitable_solution"].map(colours),
        markersize=2,
    )

    # mapping polygons of labelled data
    if tech != "District heat network":
        labelled_tech_specific_gdf = labelled_tech[labelled_tech["Name"] == tech]
        labelled_tech_specific_gdf.plot(
            ax=ax,
            column="Name",
            categorical=True,
            legend=True,
            color=labelled_tech_specific_gdf["Name"].map(colours),
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
    ward_gdf: gpd.GeoDataFrame,
    specific_ward_gdf: gpd.GeoDataFrame,
    labelled_tech: gpd.GeoDataFrame,
    colours: dict,
    threshold=None,
):
    """
    Maps properties suitable for a specific technology where the colour indicates the confidence in the block of flats label.

    Args:
        tech (str): a low carbon heating solution in the set {"Individual solution", "Networked GSHP", "Communal solutions", "District heat network"}
        ward_gdf (gpd.GeoDataFrame): ward data with most suitable solutions
        specific_ward_gdf (gpd.GeoDataFrame): specific ward boundary data
        labelled_tech (gpd.GeoDataFrame): labelled technology polygons data
        colours (dict): mapping of technologies to colours
        threshold (float, optional): threshold for the block of flats label confidence.
    """

    if tech != "":
        tech_specific_gdf = ward_gdf[ward_gdf["1st_most_suitable_solution"] == tech]
    else:
        tech_specific_gdf = ward_gdf.copy()
    fig, ax = plt.subplots(figsize=(15, 8))

    if threshold:
        tech_specific_gdf = tech_specific_gdf[
            tech_specific_gdf["block_of_flats_label_proba"] <= threshold
        ]

    print("tech:", tech)

    # mapping polygons of labelled data
    if tech != "District heat network":
        labelled_tech_specific_gdf = labelled_tech[labelled_tech["Name"] == tech]
        labelled_tech_specific_gdf.plot(
            ax=ax,
            column="Name",
            categorical=True,
            legend=True,
            color=labelled_tech_specific_gdf["Name"].map(colours),
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

    # Create a new map using only the range from 0.2 (light gray) to 1.0 (black)
    # 0.0 would be white, so we skip it.
    cmap_no_white = mcolors.LinearSegmentedColormap.from_list(
        "trunc_greys", greys(np.linspace(0.2, 1.0, 256))
    )

    # mapping individual properties where the colour indicates probability of being a block of flats
    tech_specific_gdf.plot(
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
    ward_gdf=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
# Mapping properties most suitable for individual solutions vs labelled data
map_suitable_tech_vs_labelled_tech(
    tech="Individual solution",
    ward_gdf=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %% [markdown]
# ### 6.2. Networked GSHP

# %%
map_blocks_of_flats_prob(
    tech="Networked GSHP",
    ward_gdf=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
map_suitable_tech_vs_labelled_tech(
    tech="Networked GSHP",
    ward_gdf=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %% [markdown]
# ### 6.3. Communal solutions

# %%
map_blocks_of_flats_prob(
    tech="Communal solutions",
    ward_gdf=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
map_suitable_tech_vs_labelled_tech(
    tech="Communal solutions",
    ward_gdf=ward_gdf,
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
    ward_gdf=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=intersection_shape,
    colours=colours,
)

map_suitable_tech_vs_labelled_tech(
    tech="District heat network",
    ward_gdf=ward_gdf,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=intersection_shape,
    colours=colours,
)

# %%
# Properties with blocks of flats confidence label lower than 0.6
map_blocks_of_flats_prob(
    tech="",
    ward_gdf=plymouth_gdf,
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
# precision and recall are likely not the best metrics as the labelled data isn't truly a "ground truth", so we can't read too much into these results

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
def map_building_techs_in_ward(
    techs_gdf: gpd.GeoDataFrame,
    col: str,
    specific_ward_gdf: gpd.GeoDataFrame,
    colours: dict,
    ward: str,
    underlying_data: str,
):
    """
    Maps buildings in a specific ward with a predicted/labelled low carbon heating solutions.

    Args:
        techs_gdf (gpd.GeoDataFrame): buildings with most suitable solutions
        col (str): column with tech solution
        specific_ward_gdf (gpd.GeoDataFrame): specific ward boundary data
        colours (dict): mapping of technologies to colours
        ward (str): name of the ward
        underlying_data (str): takes "Labelled" for labelled data and "Predicted" for outputs of the decision tree. Used for title and legend.
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
    ax.set_title(f"{underlying_data} most suitable solution for {ward}", fontsize=12)
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
    underlying_data="Predicted",
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
    underlying_data="Labelled",
)

# %%


# %% [markdown]
# ## 8. Visualising results per building in Plymouth

# %%
plymouth_gdf.columns

# %%
# Estimated most suitable
plymouth_building_most_suitable_tech = (
    plymouth_gdf[
        [
            "building_geometry",
            "1st_most_suitable_solution",
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
plymouth_building_most_suitable_tech["max_contiguous_outdoor_space_area_m2"] = (
    plymouth_building_most_suitable_tech["max_contiguous_outdoor_space_area_m2"].round()
)
plymouth_building_most_suitable_tech["percent_flats"] = (
    plymouth_building_most_suitable_tech["property_type_flat"]
    / plymouth_building_most_suitable_tech["UPRN"]
) * 100

# Convert back to a GeoDataFrame
plymouth_building_most_suitable_tech = gpd.GeoDataFrame(
    plymouth_building_most_suitable_tech, geometry="geometry", crs=plymouth_gdf.crs
)

# %%
plymouth_building_most_suitable_tech

# %%
map_building_techs_in_ward(
    techs_gdf=plymouth_building_most_suitable_tech,
    col="1st_most_suitable_solution",
    specific_ward_gdf=plymouth_la_boundaries_gdf,
    colours=colours,
    ward="Plymouth",
    underlying_data="Predicted",
)

# %%
# Save most suitable tech per building locally as kml
plymouth_building_most_suitable_tech[
    ["geometry", "1st_most_suitable_solution"]
].to_file("plymouth_building_most_suitable_tech.kml", driver="KML")

# %%
# Checking whether there are missing geometries (if there is no building footprint available)
plymouth_building_most_suitable_tech[
    pd.isnull(plymouth_building_most_suitable_tech["geometry"])
]

# %%
# Removing rows with no geometry, if applicable
plymouth_building_most_suitable_tech = plymouth_building_most_suitable_tech[
    ~pd.isnull(plymouth_building_most_suitable_tech["geometry"])
]

# %%
# Adding colour column for visualisation purposes
plymouth_building_most_suitable_tech["color"] = plymouth_building_most_suitable_tech[
    "1st_most_suitable_solution"
].map(colours)

# %%
plymouth_building_most_suitable_tech.columns

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

# 4. Add polygons using the color col
folium.GeoJson(
    plymouth_building_most_suitable_tech,
    style_function=lambda feature: {
        "fillColor": feature["properties"]["color"],
        "color": feature["properties"]["color"],
        "weight": 0.5,
        "fillOpacity": 1,  # Opacity
    },
    popup=popup,
).add_to(map)

map.save(
    os.path.join(
        PROJECT_DIR,
        "outputs",
        "figures",
        "plymouth_most_suitable_tech_per_building.html",
    )
)

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
        "Number of UPRNs in building:",
        "Percent of flats (%):",
        "UPRNs in block of flats:",
        "Avg max outdoor space area (m2):",
        "In city centre:",
        "In HN zone:",
        "Most suitable solution:",
    ],
    localize=True,
    labels=True,
    style="background-color: white;",
)

# 4. Add polygons using the color col
folium.GeoJson(
    gdf_plymouth_cc,
    style_function=lambda feature: {
        "fillColor": feature["properties"]["color"],
        "color": feature["properties"]["color"],
        "weight": 0.5,
        "fillOpacity": 1,  # Opacity
    },
    popup=popup,
).add_to(map_plymouth_cc)

map_plymouth_cc.save(
    os.path.join(
        PROJECT_DIR,
        "outputs",
        "figures",
        "plymouthcc_most_suitable_tech_per_building.html",
    )
)

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
# Save to GeoJSON
gdf_plymouth_cc.to_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_building_most_suitable_tech.geojson",
    driver="GeoJSON",
)

# %%
plymouth_garden_sizes_data = plymouth_building_most_suitable_tech.copy()
plymouth_garden_sizes_data["max_contiguous_outdoor_space_area_m2"].fillna(
    -100, inplace=True
)

import branca.colormap as cm

# 2. Initialize the map
# Centering based on the average coordinates of the polygons
map_garden_sizes = folium.Map(
    location=[
        plymouth_garden_sizes_data.geometry.centroid.y.mean(),
        plymouth_garden_sizes_data.geometry.centroid.x.mean(),
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

# Create a linear colormap
# Note: You can change 'YlGn' to 'Viridis', 'RdYlGn', etc.
bins = [-100, 0, 30, 100, 1000, 10000, 100000]

#
colors = [
    "#636363",  # -100 to 0 (Dark Grey - stands out from map)
    "#ffeda0",  # 0 to 30 (Pale Yellow)
    "#feb24c",  # 30 to 100 (Orange)
    "#fc4e2a",  # 100 to 1000 (Bright Red)
    "#bd0026",  # 1000 to 10000 (Deep Crimson)
    "#800026",  # Above 10000 (Dark Maroon)
]

colormap = cm.StepColormap(
    colors=colors, index=bins, vmin=-100, vmax=bins[-1], caption="Garden Size (m2)"
)

# Add the colormap legend to the map
colormap.add_to(map_garden_sizes)

# 4. Add polygons using the color col
folium.GeoJson(
    plymouth_garden_sizes_data,
    style_function=lambda feature: {
        "fillColor": colormap(
            feature["properties"]["max_contiguous_outdoor_space_area_m2"]
        ),
        "color": colormap(
            feature["properties"]["max_contiguous_outdoor_space_area_m2"]
        ),
        "weight": 0.5,
        "fillOpacity": 1,  # Opacity
    },
    popup=popup,
).add_to(map_garden_sizes)


map_garden_sizes.save(
    os.path.join(PROJECT_DIR, "outputs", "figures", "plymouth_garden_sizes.html")
)

# %%
plymouth_garden_sizes_data = plymouth_building_most_suitable_tech.copy()
plymouth_garden_sizes_data["max_contiguous_outdoor_space_area_m2"].fillna(
    -100, inplace=True
)
plymouth_garden_sizes_data = plymouth_garden_sizes_data[
    plymouth_garden_sizes_data["max_contiguous_outdoor_space_area_m2"] == -100
]


# 4. Add polygons using the color col
folium.GeoJson(
    plymouth_garden_sizes_data,
).add_to(map_garden_sizes)


map_garden_sizes.save(
    os.path.join(PROJECT_DIR, "outputs", "figures", "plymouth_garden_sizes_2.html")
)

# %%


# %%
