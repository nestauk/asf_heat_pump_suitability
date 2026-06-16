# %% [markdown]
# # Calculate threshold for nearest neighbour search
#
# Scotland and Wales listed buildings data is available as point geometries rather than polygons. We also have point geometries for UPRNs in the EPC dataset. In order to join EPC UPRNs to listed buildings, we need to use a nearest neighbour search. This notebook uses ground-truth listed building polygon data for Scotland to identify the threshold of distance in metres for determining whether a UPRN is located within a listed building.

# %%
import geopandas as gpd
import polars as pl
import matplotlib.pyplot as plt

from asf_heat_pump_suitability.pipeline.transform import lat_lon

# %%
# Listed building point geoms Scotland
points_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/source_data/lb_scotland/Listed_Buildings.shp"
)
points_gdf = points_gdf[["ENT_REF", "ENT_TITLE", "geometry"]]

# %%
epc = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/20240904_2023_Q4_EPC_weighted_features_gardens.parquet"
)

# %%
epc_gdf = epc.filter(pl.col("COUNTRY") == "Scotland")
epc_gdf = lat_lon.generate_gdf_uprn_coords(epc_gdf, usecols=["UPRN", "COUNTRY"])

# %% [markdown]
# ## Compare to ground truth

# %%
# Listed building boundaries for Scotland (limited dataset)
boundaries = gpd.read_file(
    "s3://asf-heat-pump-suitability/source_data/lb_scotland/Listed_Buildings_boundaries.shp"
)

# %%
# Join listed building polygons to listed building points
boundaries = boundaries[["DES_REF", "DES_TITLE", "geometry"]]
ground_truth = boundaries.sjoin(points_gdf, how="inner", predicate="intersects").drop(
    columns=["index_right"]
)

# %%
# Join listed buildings to EPC using polygons to identify true matches
gdf = epc_gdf.sjoin(
    boundaries[boundaries["DES_REF"].isin(ground_truth["DES_REF"])],
    how="left",
    predicate="intersects",
).drop(columns=["index_right"])

# Calculate distance from nearest listed building for each EPC record
gdf = gdf.sjoin_nearest(
    points_gdf[points_gdf["ENT_REF"].isin(ground_truth["ENT_REF"])],
    how="left",
    max_distance=500,
    distance_col="distance_from_nearest_listed_m",
)
df = gdf.drop(columns=["geometry"])

df = pl.from_pandas(df)
df = df.with_columns(
    pl.when(pl.col("DES_REF").is_not_null())
    .then(True)
    .otherwise(False)
    .alias("true_match")
)
df = df.filter(pl.col("distance_from_nearest_listed_m").is_not_null())
df.head()

# %%
# Visualise distance for true matches vs non-matches
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].boxplot(df.filter(pl.col("true_match"))["distance_from_nearest_listed_m"])
axs[0].set_title("True matches")
axs[0].set_ylabel("Distance from nearest listed building (m)")

axs[1].boxplot(df.filter(~pl.col("true_match"))["distance_from_nearest_listed_m"])
axs[1].set_title("Not matches")
plt.suptitle("Distance from nearest listed building point geom (m)")


# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5))

axs.boxplot(
    df.filter(pl.col("true_match"), pl.col("distance_from_nearest_listed_m") <= 20)[
        "distance_from_nearest_listed_m"
    ]
)
axs.set_title("True matches")
axs.set_ylabel("Distance from nearest listed building (m)")

# %%
df.filter(pl.col("true_match"))["distance_from_nearest_listed_m"].describe()

# %% [markdown]
# ## Test threshold distance

# %%
test = epc_gdf.sjoin_nearest(
    points_gdf,
    how="inner",
    max_distance=5,
    distance_col="distance_from_nearest_listed_m",
)

# %%
test.shape
