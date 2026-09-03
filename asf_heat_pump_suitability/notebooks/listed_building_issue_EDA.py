# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: asf_heat_pump_suitability
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### ISSUE 168
#
# Liz noticed there are 61 listed buildings in the dataset for the area of Stoke in Plymouth. This is the case for both:
# - The listed building dataset currently in our config s3://asf-heat-pump-suitability/source_data/Jun2024_vJul2024_HistoricEngland_listedbuilding_E.gpkg
# - New data I got from the Listed Building Historic England website
# (So this issue isn't because of the need for a data refresh.)
#
# However, when you look at the properties which have is_listed=True for this region, there are only 6. (using s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250319_2023_Q4_heat_pump_suitability_per_property.parquet)
#
# Liz plotted the properties we have in this per_property dataset and the 61 listed buildings here. You can see that some of the listed buildings (blue) do overlap with EPC properties so it seems weird these properties weren't given is_listed=True.
#
# ![alt text](image.png)

# %% [markdown]
# ### Load data
#
# Load the suitability per property dataset as well as the boundary data for LSOAs and Wards in the UK

# %%
import pandas as pd
import polars as pl

# %%
import s3fs

fs = s3fs.S3FileSystem()
import geopandas as gpd

# %%
# The boundaries of each LSOA for plotting
lsoa_boundaries_file = "s3://asf-heat-pump-suitability/source_data/Lower_layer_Super_Output_Areas_2021_EW_BFE_V9_-9107090204806789093/LSOA_2021_EW_BFE_V9.shp"

# Suitability data per LSOA
suitablitity_per_lsoa_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250319_2023_Q4_heat_pump_suitability_per_lsoa.parquet"

# The lat/long coordinates of properties for plotting
# lat_long_property_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/features/20250110_2023_Q4_EPC_features.parquet"

# The suitability of properties
suitability_per_property_file = "s3://asf-heat-pump-suitability/outputs/2023Q4/suitability/20250319_2023_Q4_heat_pump_suitability_per_property.parquet"

# %%
# https://www.data.gov.uk/dataset/0bdfd7a6-e6a4-4d63-a684-b6dda1d86d47/wards-may-2024-boundaries-uk-bsc

wards_boundaries_file = "s3://asf-heat-pump-suitability/source_data/Wards_May_2024_Boundaries_UK_BSC_-6022693383694477628.geojson"

# %% [markdown]
# ### Find which LSOAs are in the Stoke ward

# %%
# In BNG EPSG: 27700
wards_boundaries = gpd.read_file(wards_boundaries_file)

# %%
stoke_ward = wards_boundaries[wards_boundaries["WD24CD"] == "E05002096"]

# %%
# In BNG EPSG: 27700
lsoa_boundaries = gpd.read_file(lsoa_boundaries_file)

# %%
stoke_boundaries = stoke_ward.sjoin(lsoa_boundaries, how="left", predicate="intersects")


# %%
stoke_lsoas = stoke_boundaries["LSOA21CD"].unique().tolist()


# %% [markdown]
# ### Just Stoke
# - Now we read in the suitability per property data in this particularly set, where we want to filter it for the LSOAs in the Stoke ward
# - We also want to filter it for the property points geometries with listed building status.

# %%
per_prop_data = pl.read_parquet(
    suitability_per_property_file,
)

# %%
area_per_prop_data = per_prop_data.filter(pl.col("lsoa").is_in(stoke_lsoas)).to_pandas()

# %%
area_per_prop_data["listed_building"].value_counts()

# %%
area_per_prop_data["is_listed"] = area_per_prop_data["listed_building"].astype(int)

# %% [markdown]
# ### Let's look at the Historic England data to get a sense of what's in there

# %%
import geopandas as gpd

# Paths
he_file = "s3://asf-heat-pump-suitability/source_data/Jun2024_vJul2024_HistoricEngland_listedbuilding_E.gpkg"

# %%
# Load ward boundaries and select Stoke
# In BNG EPSG: 27700
wards_boundaries = gpd.read_file(wards_boundaries_file)
stoke_ward = wards_boundaries[wards_boundaries["WD24CD"] == "E05002096"]


# %%
# Load HE listed buildings
# In BNG EPSG: 27700 and also geometry is MultiPolygon
he_listed = gpd.read_file(he_file)
print(he_listed.geometry.geom_type.value_counts())


# %%
he_listed.head()

# %%
# Spatial filter: listed buildings that intersect the Stoke Ward
he_stoke_listed = gpd.sjoin(he_listed, stoke_ward, how="inner", predicate="intersects")


# %%
he_stoke_listed.columns

# %%
print(f"Listed buildings in Stoke Ward: {len(he_stoke_listed)}")

# %%
area_per_prop_data.columns

# %% [markdown]
# ### Comparison of the HE england data with the EPC/suitability dataset in terms of the Listed Buildings geometries.

# %%
# EPC points from BNG
epc_stoke_properties_gdf = gpd.GeoDataFrame(
    area_per_prop_data,
    geometry=gpd.points_from_xy(
        area_per_prop_data["X_COORDINATE"], area_per_prop_data["Y_COORDINATE"]
    ),
    crs=27700,
)


# %%
epc_stoke_properties_gdf.columns

# %%
# Subsets
epc_all = epc_stoke_properties_gdf
pipeline_listed = epc_all[
    epc_all["is_listed"] == 1
]  # or epc_all["listed_building"] == True
he_listed_stoke = he_stoke_listed  # already filtered to ward

# Basic overlap diagnostics
overlap = gpd.sjoin(
    pipeline_listed[["geometry"]],
    he_listed_stoke[["geometry"]],
    how="inner",
    predicate="intersects",
)
print(
    f"EPC all: {len(epc_all)} | Pipeline listed: {len(pipeline_listed)} | HE listed: {len(he_listed_stoke)} | Overlap: {len(overlap)}"
)


# %% [markdown]
# ### We want to now visualise/plot to get a sense of what this looks like on a map

# %%
# !pip install contextily

# %%
# pip install contextily
import contextily as ctx
import matplotlib.pyplot as plt

# Plot (convert to WGS84 for nicer map-like view)
epc_all_plot = epc_all.to_crs(4326)
pipeline_listed_plot = pipeline_listed.to_crs(4326)
he_plot = he_listed_stoke.to_crs(4326)
ward_plot = stoke_ward.to_crs(4326)


# 1) Reproject EVERYTHING to Web Mercator (meters) for web tiles
ward_web = stoke_ward.to_crs(3857)  # EPSG:3857 is Web Mercator
# ward_web      = ward_plot.to_crs(3857)
he_web = he_plot.to_crs(3857)
pipeline_web = pipeline_listed_plot.to_crs(3857)
overlap_web = overlap.to_crs(3857)
# epc_all_plot if you want it too:
epc_web = epc_all_plot.to_crs(3857)

# 2) Plot your layers
xmin, ymin, xmax, ymax = ward_web.total_bounds
fig, ax = plt.subplots(figsize=(9, 9))

ward_web.plot(
    ax=ax, color="#f5c7c7", alpha=0.25, edgecolor="black", linewidth=1, zorder=2
)
he_web.boundary.plot(ax=ax, edgecolor="blue", linewidth=2, alpha=0.9, zorder=3)
pipeline_web.plot(ax=ax, color="red", markersize=14, alpha=0.6, zorder=4)
overlap_web.plot(
    ax=ax,
    facecolor="none",  # no fill
    edgecolor="purple",
    marker="o",
    markersize=40,  # adjust as needed
    linewidth=1.2,
    zorder=5,
)
epc_web.plot(ax=ax, color="grey", markersize=6, alpha=0.55, zorder=1)  # optional

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# 3) Add basemap (choose a provider you like)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)  # or OpenStreetMap.Mapnik
ax.axis("off")
plt.savefig("stoke_ward_listed_buildings.pdf", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()


# Also confirm all v

# %% [markdown]
# #### Hypothesis about the issue
# - EPC points are actually outside the Listed Buildings polygons and therefore are not being picked up by the spatial join in the Listed Buildings algorithm (tbc).
# - Potentialy solution is to add a (small) buffer zone around the Listed Buildings algorithm to pick up EPC points.
# - Danger is that we then have false positives because of the above change.
#
# In the below python, we essentilly try out this solution and then manually validate some of the 'new positives' (those that fall in the newly created 5 m zone) to see where they are actually listed buildings.

# %%
### This is essentially the function we use to classify listed buildings for our Suitability dataset. We've now added in a buffer parameter for the polygon part of the if statement.


def sjoin_df_epc_listed_buildings(
    suitability_df: pl.DataFrame,
    listed_buildings_gdf: gpd.GeoDataFrame,
    distance: float = 10,
    buffer_m: float = 0,
) -> pd.DataFrame:
    """
    Spatial join EPC UPRNs with listed buildings using `geopandas.GeoDataFrame.sjoin_nearest` where Point or MultiPoint
    geometries detected, and `geopandas.GeoDataFrame.sjoin` where Polygons or MultiPolygons detected.

    Args:
        suitability_df (pl.DataFrame): suitability dataset with UPRNs and geometries
        listed_buildings_gdf (gpd.GeoDataFrame): listed buildings data
        distance (float): maximum distance (m) within which to query for nearest geometry where `sjoin_nearest` used.
                          Must be greater than 0. Default 10m.

    Returns:
        pd.DataFrame: EPC UPRNs in listed buildings
    """
    # epc_gdf = lat_lon.generate_gdf_uprn_coords(df=epc_df, usecols=["UPRN"])
    if any(
        [
            expr in listed_buildings_gdf.geom_type.unique()
            for expr in ["Point", "MultiPoint"]
        ]
    ):
        df = suitability_df.sjoin_nearest(
            listed_buildings_gdf, how="inner", max_distance=distance
        )[["UPRN", "listed_building"]].drop_duplicates(subset="UPRN")
    elif any(
        [
            expr in listed_buildings_gdf.geom_type.unique()
            for expr in ["Polygon", "MultiPolygon"]
        ]
    ):
        poly = listed_buildings_gdf
        if buffer_m > 0:
            poly = poly.copy()
            poly["geometry"] = poly.geometry.buffer(buffer_m)
        df = suitability_df.sjoin(poly, how="inner", predicate="intersects")[
            ["UPRN", "listed_building"]
        ].drop_duplicates("UPRN")
    else:
        raise ValueError(
            f"Listed buildings GeoDataFrame does not have appropriate geometries for sjoin. "
            f"Geometries required: [Multi]Point or [Multi]Polygon. "
            f"Geometries found: {listed_buildings_gdf.geom_type.unique()}"
        )
    return df


# %%
# For an initial example, we can add in a 5 metre buffer zone
he_listed_stoke_clean = he_listed_stoke.drop(columns=["index_right"], errors="ignore")

matches = sjoin_df_epc_listed_buildings(
    suitability_df=epc_stoke_properties_gdf,
    listed_buildings_gdf=he_listed_stoke_clean,
    distance=10,
    buffer_m=5,
)


# %%
matches.head()


# %%
# Unique UPRNs that hit original+2m buffer
m_ids = matches[["UPRN"]].drop_duplicates()["UPRN"].astype(str)

epc = epc_stoke_properties_gdf
epc["is_listed_5m"] = epc["UPRN"].astype(str).isin(m_ids).astype("int8")

# %%
print("5m total:", int(epc["is_listed_5m"].sum()))

# %%
epc_stoke_properties_gdf.columns


# %%
he_listed_stoke.columns
print(he_listed_stoke.geometry.geom_type.value_counts())

# %%
# Subsets
epc_all = epc
pipeline_listed = epc_all[
    epc_all["is_listed_5m"] == 1
]  # or epc_all["listed_building"] == True
he_listed_stoke = he_stoke_listed  # already filtered to ward
# epc_all.head()

# %% [markdown]
# ### Plotting the overlaps of EPC listed building properties and those found in the Historic England dataset to see how this new algorithm performs

# %%
# Buffered polygons (2 m)
he_buf_5m = he_listed_stoke.copy()
he_buf_5m["geometry"] = he_buf_5m.geometry.buffer(5)
overlap = gpd.sjoin(
    pipeline_listed[["geometry"]],
    he_buf_5m[["geometry"]],
    how="inner",
    predicate="intersects",
)
print(
    f"EPC all: {len(epc_all)} | Pipeline listed: {len(pipeline_listed)} | HE listed: {len(he_listed_stoke)} | Overlap: {len(overlap)}"
)

# Plot (convert to WGS84 for nicer map-like view)
epc_all_plot = epc_all.to_crs(4326)
pipeline_listed_plot = pipeline_listed.to_crs(4326)
he_plot = he_buf_5m.to_crs(4326)
ward_plot = stoke_ward.to_crs(4326)

# 1) Reproject EVERYTHING to Web Mercator (meters) for web tiles
ward_web = stoke_ward.to_crs(3857)  # EPSG:3857 is Web Mercator
# ward_web      = ward_plot.to_crs(3857)
he_web = he_plot.to_crs(3857)
pipeline_web = pipeline_listed_plot.to_crs(3857)
overlap_web = overlap.to_crs(3857)
# epc_all_plot if you want it too:
epc_web = epc_all_plot.to_crs(3857)

# 2) Plot your layers
xmin, ymin, xmax, ymax = ward_web.total_bounds
fig, ax = plt.subplots(figsize=(9, 9))

ward_web.plot(
    ax=ax, color="#f5c7c7", alpha=0.25, edgecolor="black", linewidth=1, zorder=2
)
he_web.boundary.plot(ax=ax, edgecolor="blue", linewidth=2, alpha=0.9, zorder=3)
pipeline_web.plot(ax=ax, color="red", markersize=14, alpha=0.6, zorder=4)
overlap_web.plot(
    ax=ax,
    facecolor="none",  # no fill
    edgecolor="purple",
    marker="o",
    markersize=40,  # adjust as needed
    linewidth=1.2,
    zorder=5,
)
epc_web.plot(ax=ax, color="grey", markersize=6, alpha=0.55, zorder=1)  # optional

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# 3) Add basemap (choose a provider you like)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)  # or OpenStreetMap.Mapnik
ax.axis("off")
plt.savefig("stoke_ward_listed_buildings_5m.pdf", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Create a validation dataset to manually check with Google maps
# - Essentially create three different buffer zones (exact-0 m, small buffer 0-2 m and then medium buffer (2-5 ms) aswell as calculating the distance). Priotise in terms of checking in terms of buildings next to each other or with quite large distances (closer to 5 metres) for evaluation checks.

# %%
# Clean HE (keep ListEntry as a column)
he_orig = he_listed_stoke.drop(columns=["index_right"], errors="ignore").copy()
he_geom_by_id = he_orig.set_index("ListEntry").geometry  # for distance lookup

# Buffered variants
he_buf_2m = he_orig.copy()
he_buf_2m["geometry"] = he_buf_2m.geometry.buffer(2)
he_buf_5m = he_orig.copy()
he_buf_5m["geometry"] = he_buf_5m.geometry.buffer(5)


def sjoin_stage(
    epc_gdf: gpd.GeoDataFrame, he_gdf: gpd.GeoDataFrame, stage_name: str
) -> pd.DataFrame:
    """
    Spatially joins EPC property points to Historic England polygons, annotating each match with metadata.

    For each EPC, finds intersecting HE polygons, computes distance to the original (unbuffered) polygon,
    and keeps only the closest match per UPRN. Adds match type and HE metadata.

    Args:
        epc_gdf: GeoDataFrame of EPC properties (must include 'UPRN', 'geometry').
        he_gdf: GeoDataFrame of HE polygons (must include 'ListEntry', 'geometry').
        stage_name: Label for the matching stage.

    Returns:
        DataFrame with columns:
            ['UPRN', 'POSTCODE', 'property_type', 'build_year',
             'match_type', 'dist_m', 'he_id', 'he_name', 'he_grade']
    """
    cols_keep = ["UPRN", "POSTCODE", "property_type", "build_year", "geometry"]
    right_cols = ["geometry", "ListEntry", "Name", "Grade"]
    res = gpd.sjoin(
        epc_gdf[cols_keep], he_gdf[right_cols], how="inner", predicate="intersects"
    )
    if res.empty:
        return res

    # Distance to original (unbuffered) HE polygon (0 if inside/touching)
    res["dist_m"] = res.apply(
        lambda r: r.geometry.distance(he_geom_by_id.loc[r["ListEntry"]]), axis=1
    )

    res["match_type"] = stage_name
    res["he_id"] = res["ListEntry"]
    res["he_name"] = res["Name"]
    res["he_grade"] = res["Grade"]

    # If multiple HE polygons per UPRN, keep the closest
    res = res.sort_values(["UPRN", "dist_m"]).drop_duplicates("UPRN", keep="first")
    return res[
        [
            "UPRN",
            "POSTCODE",
            "property_type",
            "build_year",
            "match_type",
            "dist_m",
            "he_id",
            "he_name",
            "he_grade",
        ]
    ]


# Run staged matching
stage_exact = sjoin_stage(epc_stoke_properties_gdf, he_orig, "exact")
remaining = epc_stoke_properties_gdf[
    ~epc_stoke_properties_gdf["UPRN"].isin(stage_exact["UPRN"])
]

stage_2m = sjoin_stage(remaining, he_buf_2m, "buffer_2m")
remaining2 = remaining[~remaining["UPRN"].isin(stage_2m["UPRN"])]

stage_5m = sjoin_stage(remaining2, he_buf_5m, "buffer_5m")

validation = pd.concat([stage_exact, stage_2m, stage_5m], ignore_index=True)

# Count EPCs per HE polygon
counts = validation.groupby("he_id")["UPRN"].nunique()
validation["uprns_per_he"] = validation["he_id"].map(counts)

print(validation["match_type"].value_counts(dropna=False))
print("Total matched EPCs:", validation["UPRN"].nunique())
validation.head()
validation.to_csv("stoke_ward_listed_buildings_matches.csv", index=False)

# %% [markdown]
# ### Another potential solution is to match EPC points in the per property suitability dataset with the building footprints provided by the OS OpenMap Local and then match with the HE listed building multipolygons. We make an initial check to see how many of the EPC points land within those multipolygons.

# %%
# Load OS building footprints
buildings = gpd.read_file("OS_OpenMap_Local_plymouth/data/SX_Building.shp").to_crs(
    epsg=27700
)

# Ensure EPC points are in same CRS
epc_points = epc_stoke_properties_gdf.to_crs(epsg=27700)

# Spatial join: keep only EPC points within a building footprint
epc_in_building = gpd.sjoin(epc_points, buildings, how="inner", predicate="within")

# Calculate percentage
pct_in_buildings = len(epc_in_building) / len(epc_points) * 100

print(f"Total EPC points: {len(epc_points)}")
print(f"EPC points inside building footprints: {len(epc_in_building)}")
print(f"Percentage inside buildings: {pct_in_buildings:.2f}%")

# %%
buildings.head()
