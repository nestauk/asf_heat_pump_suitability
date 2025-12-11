# %% [markdown]
# # Exploring visualising the results of the decision tree

# %%
# package imports
import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %%
# local imports
from asf_heat_pump_suitability.pipeline.transform.uprns import generate_gdf_uprn_coords
from asf_heat_pump_suitability.getters.load_tree_input import (
    load_gdf_os_openmap_local_layer,
)
from asf_heat_pump_suitability.getters.load_geodata import (
    load_gdf_heat_network_zones,
)

# %% [markdown]
# ## 1. Loading data

# %% [markdown]
# ### 1.1. Manually labelled data for one specific ward

# %%
labelled_tech = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/plymouth_spcific_ward_labelled_technology_polygons.kml"
)

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
labelled_tech = labelled_tech.to_crs(epsg=27700)

# %% [markdown]
# ### 1.2. Getting Plymouth residential UPRNs data
# And converting df to geodf

# %%
plymouth_uprns = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns.parquet"
)
plymouth_uprns = generate_gdf_uprn_coords(df=plymouth_uprns)
plymouth_uprns.head()

# %% [markdown]
# ### 1.3. Getting info about city centres and HNZ

# %%
city_centre_data = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_city_centres.parquet"
)
city_centre_data = generate_gdf_uprn_coords(df=city_centre_data)
city_centre_data.head()

# %%
hnz_data = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_hn_zones.parquet"
)
hnz_data = generate_gdf_uprn_coords(df=hnz_data)
hnz_data.head()

# %% [markdown]
# ### 1.4. Loading Plymouth features dataset

# %%
plymouth_with_features = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_features.parquet"
)
plymouth_with_features = generate_gdf_uprn_coords(df=plymouth_with_features)
plymouth_with_features.head()

# %% [markdown]
# ### 1.5. Loading building footprints

# %%
# building footprints (still needs to be filtered for plymouth)
building_footprints = load_gdf_os_openmap_local_layer(
    layer="building", grid_squares="SX"
)
building_footprints.head()

# %% [markdown]
# ### 1.6. Loading specific ward

# %%
specific_ward_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/source_data/plymouth_specific_ward_boundary.geojson"
)

# %%
specific_ward_gdf.crs

# %%
specific_ward_gdf.to_crs(epsg=27700, inplace=True)

# %%
specific_ward_gdf.head()

# %%
ward = specific_ward_gdf.name.values[0]

# %% [markdown]
# ### 1.7. Load HN zones

# %%
plymouth_hn_zones_gdf = load_gdf_heat_network_zones(local_authority="plymouth")

# %%
plymouth_hn_zones_gdf.crs

# %%
plymouth_hn_zones_gdf.head()

# %% [markdown]
# ## 2. Processing data

# %% [markdown]
# ### 2.1. [temporary] Joining all dfs into one geodf

# %%
plymouth_uprns.set_index("UPRN", inplace=True)
plymouth_with_features.set_index("UPRN", inplace=True)
city_centre_data.set_index("UPRN", inplace=True)
hnz_data.set_index("UPRN", inplace=True)

# %%
plymouth_uprns.head()

# %%
len(plymouth_uprns), len(plymouth_with_features), len(city_centre_data), len(hnz_data)

# %%
# Joining everything into the same df
plymouth_df = plymouth_uprns.join(city_centre_data[["in_city_centre"]])
plymouth_df = plymouth_df.join(hnz_data[["in_hn_zone"]])
plymouth_df = plymouth_df.join(
    plymouth_with_features.drop(columns=["X_COORDINATE", "Y_COORDINATE", "geometry"])
)

# %%
plymouth_df.reset_index(inplace=True)

# %% [markdown]
# ### 2.2. [temporary] Identifying properties in blocks of flats

# %%
# counts of properties per building
counts = plymouth_df.groupby("NATIONALCADASTRALREFERENCE").count()


## Roisin's suggestion (I ended up doing something easier, based on UPRN counts per land parcel)
# groupby NATIONALCADASTRALREFERENCE (land parcel) - if garden size > Xm2 and Y number of UPRNs then block (for now)

# NATIONALCADASTRALREFERENCE: ID of the land parcel
# A land extent should represent a section of a building, rather than the whole building footprint
# every uprn that shares a garden is in the same land extent

# %%
# While we wait for the modelled data, we identify blocks of flats as those with 6 or more UPRNs per land parcel
blocks_of_flats_ncref = counts[counts["UPRN"] >= 6].index.tolist()

plymouth_df["is_in_block_of_flats"] = plymouth_df["NATIONALCADASTRALREFERENCE"].apply(
    lambda x: x in blocks_of_flats_ncref
)

# %% [markdown]
# ### 2.3. Joining building footprints to main geodf

# %%
plymouth_df["property_geometry"] = plymouth_df.geometry
building_footprints["building_geometry"] = building_footprints["geometry"]
plymouth_df = building_footprints[["geometry", "building_geometry"]].sjoin(
    plymouth_df, how="right", predicate="intersects"
)

# %% [markdown]
# ### 2.4. sjoin with specific ward

# %%
plymouth_df.drop(columns=["index_left"], inplace=True)

# %%
specific_ward_gdf["ward"] = ward
plymouth_df = plymouth_df.sjoin(
    specific_ward_gdf[["ward", "geometry"]], how="left", predicate="within"
)

# %% [markdown]
# ### 2.5. sjoin with DESNZ HN zones

# %%
plymouth_hn_zones_gdf["desnz_hn_zone"] = True
plymouth_hn_zones_gdf["desnz_hn_zone_geometry"] = plymouth_hn_zones_gdf["geometry"]

# %%
plymouth_df.drop(columns=["index_right"], inplace=True)

# %%
plymouth_df = plymouth_df.sjoin(
    plymouth_hn_zones_gdf[["desnz_hn_zone", "geometry", "desnz_hn_zone_geometry"]],
    how="left",
    predicate="within",
)
plymouth_df["desnz_hn_zone"].fillna(False, inplace=True)

# %% [markdown]
# ### 2.6. Replacing any labelled data in HN zones (as one of the other solutions) as district HN
# Some areas that are covered by HN zones have been manually labelled as not HN, so we need to replace those labels.

# %%
labelled_tech = labelled_tech.sjoin(
    plymouth_hn_zones_gdf[["desnz_hn_zone", "geometry", "desnz_hn_zone_geometry"]],
    how="left",
    predicate="intersects",
)

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
    in_block_of_flats: bool, garden_size: float, city_centre: bool, hn_zone: bool
) -> dict:
    """
    Defines the decision tree to identify first and second most suitable low carbon heating solutions for each property.

    Args:
        in_block_of_flats (bool): Whether the property is in a block of flats.
        garden_size (float): Size of the garden in square meters.
        city_centre (bool): Whether the property is in the city centre.
        hn_zone (bool): Whether the property is in a heat network zone.

    Returns:
        dict: A dictionary with the first and second most suitable heating solutions.
    """
    city_centre_or_hnz = city_centre or hn_zone

    if in_block_of_flats:
        if city_centre_or_hnz:
            return {1: "District heat network", 2: "Communal solutions"}
        else:
            return {1: "Communal solutions", 2: "Networked GSHP"}
    else:
        if city_centre_or_hnz:
            if garden_size > 70:
                return {1: "Individual solution", 2: "District heat network"}
            else:
                return {1: "District heat network", 2: "Networked GSHP"}
        else:
            if garden_size > 30:
                return {1: "Individual solution", 2: "Networked GSHP"}
            else:
                return {1: "Networked GSHP", 2: "Communal solutions"}


# %%
plymouth_df["most_suitable_solutions"] = plymouth_df.apply(
    lambda x: define_decision_tree(
        x["is_in_block_of_flats"],
        x["max_contiguous_outdoor_space_area_m2"],
        x["in_city_centre"],
        x["in_hn_zone"],
    ),
    axis=1,
)

# %%
plymouth_df["1st_most_suitable_solution"] = plymouth_df[
    "most_suitable_solutions"
].apply(lambda x: x[1])
plymouth_df["2nd_most_suitable_solution"] = plymouth_df[
    "most_suitable_solutions"
].apply(lambda x: x[2])

# %% [markdown]
# ### 4. Visualising "predictions" against labelled data in one ward

# %%
colours = {
    "Individual solution": "orange",
    "Networked GSHP": "green",
    "Communal solutions": "hotpink",
    "District heat network": "blue",
}

# %%
ward_df = plymouth_df[plymouth_df["ward"] == ward]


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
    ax.set_title(
        f"Properties in {ward} ward most suitable for {tech.lower()}", fontsize=12
    )
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
map_suitable_tech_vs_labelled_tech(
    tech="Individual solution",
    ward_df=ward_df,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
map_suitable_tech_vs_labelled_tech(
    tech="Networked GSHP",
    ward_df=ward_df,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
map_suitable_tech_vs_labelled_tech(
    tech="Communal solutions",
    ward_df=ward_df,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=labelled_tech,
    colours=colours,
)

# %%
ward_hnz_join = plymouth_hn_zones_gdf.sjoin(
    specific_ward_gdf, how="inner", predicate="intersects"
)
intersection_shape = gpd.overlay(ward_hnz_join, specific_ward_gdf, how="intersection")

map_suitable_tech_vs_labelled_tech(
    tech="District heat network",
    ward_df=ward_df,
    specific_ward_gdf=specific_ward_gdf,
    labelled_tech=intersection_shape,
    colours=colours,
)

# %% [markdown]
# ## 5. Assessing the results of the decision tree for the labelled data in one ward

# %%
labelled_ward_stats = ward_df[["geometry", "1st_most_suitable_solution"]].sjoin(
    labelled_tech[["Name", "geometry"]], how="left", predicate="within"
)


# %%
labelled_ward_stats = labelled_ward_stats.drop(columns=["index_right"]).rename(
    columns={
        "Name": "labelled_tech",
        "1st_most_suitable_solution": "most_suitable_solution",
    }
)

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
# ## 6. Number of solutions per building

# %%
# Each building/land parcel should ideally have only one most suitable solution
ward_df.groupby("NATIONALCADASTRALREFERENCE")[["1st_most_suitable_solution"]].nunique()[
    "1st_most_suitable_solution"
].value_counts()

# %% [markdown]
# ## 7. Visualising results per building in the labelled ward

# %%
ward_building_most_suitable_tech = (
    ward_df[["building_geometry", "1st_most_suitable_solution"]]
    .drop_duplicates()
    .rename(columns={"building_geometry": "geometry"})
)


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
    ax.set_title(f"{map_} most suitable solution for the {ward} ward", fontsize=12)
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
ward_building_labelled_tech = (
    ward_df[["building_geometry", "geometry"]]
    .sjoin(labelled_tech[["Name", "geometry"]], how="left", predicate="within")
    .drop(columns=["geometry", "index_right"])
    .drop_duplicates()
    .rename(columns={"building_geometry": "geometry", "Name": "labelled_tech"})
)

# %%
# not sure why some buildings don't have a labelled tech, need to investigate!
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

# %% [markdown]
# ## 8. Visualising results per building in Plymouth

# %%
plymouth_building_most_suitable_tech = (
    plymouth_df[["building_geometry", "1st_most_suitable_solution"]]
    .drop_duplicates()
    .rename(columns={"building_geometry": "geometry"})
)


# %%
from asf_heat_pump_suitability.getters import (
    load_boundaries,
)

# Plymouth LA boundary
plymouth_la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
    select_las="plymouth"
)

# %%
map_building_techs_in_ward(
    techs_gdf=plymouth_building_most_suitable_tech,
    col="1st_most_suitable_solution",
    specific_ward_gdf=plymouth_la_boundaries_gdf,
    colours=colours,
    ward=ward,
    map_="Predicted",
)

# %% [markdown]
# ## 9. Clustering buildings in specific ward

# %%
ward_building_most_suitable_tech["x"] = (
    ward_building_most_suitable_tech.geometry.centroid.x
)
ward_building_most_suitable_tech["y"] = (
    ward_building_most_suitable_tech.geometry.centroid.y
)
df_encoded = pd.get_dummies(
    ward_building_most_suitable_tech,
    columns=["1st_most_suitable_solution"],
    drop_first=True,
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_features = scaler.fit_transform(df_encoded.drop(columns=["geometry"]))

# %%
# clustering buildings by most suitable tech and location using HDBSCAN
from sklearn.cluster import HDBSCAN
