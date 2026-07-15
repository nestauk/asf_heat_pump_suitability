# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    root_mean_squared_log_error,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans  #

# %%
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata
from asf_heat_pump_suitability.pipeline.transform import local_authority
from asf_heat_pump_suitability.pipeline.cluster import cluster


# %%
def _get_gdf_5nn_spatial_features(gdf, unique_id_col):
    """
    Takes a GeoDataFrame of buildings, finds the 5 nearest neighbors that have
    known garden sizes, and extracts their individual sizes and distances.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame of buildings footprints or UPRN point coordinates
        unique_id_col (str): name of ID column

    Returns:
        gpd.GeoDataFrame: original gdf with columns for 5 nearest neighbour garden sizes and distances
    """
    # get centroids and IDs of uprns/ buildings
    all_centroids = np.column_stack((gdf.geometry.centroid.x, gdf.geometry.centroid.y))
    all_ids = gdf[unique_id_col].values

    # get just the known garden sizes and the centroids of them
    known_gardens = gdf.dropna(subset=["max_contiguous_outdoor_space_area_m2"])
    known_centroids = np.column_stack(
        (known_gardens.geometry.centroid.x, known_gardens.geometry.centroid.y)
    )
    known_sizes = known_gardens["max_contiguous_outdoor_space_area_m2"].values
    known_ids = known_gardens[unique_id_col].values

    # find the 6 nearest garden sizes and the distance to them. We need 6 so that we can drop the self-match later
    nn = NearestNeighbors(n_neighbors=6, algorithm="kd_tree")
    nn.fit(known_centroids)
    distances, indices = nn.kneighbors(all_centroids)

    # Initialize storage lists
    nn1_size, nn2_size, nn3_size, nn4_size, nn5_size = [], [], [], [], []
    nn1_dist, nn2_dist, nn3_dist, nn4_dist, nn5_dist = [], [], [], [], []

    # Loop through the array
    for i in range(len(all_centroids)):
        target_id = all_ids[i]
        row_indices = indices[i]
        row_distances = distances[i]

        # drop the self-match
        neighbor_ids = known_ids[row_indices]
        valid_mask = neighbor_ids != target_id

        valid_sizes = known_sizes[row_indices[valid_mask]]
        valid_distances = row_distances[valid_mask]

        sizes_to_use = list(valid_sizes[:5])
        dists_to_use = list(valid_distances[:5])

        # Pad with NaNs if there are extremely isolated buildings
        while len(sizes_to_use) < 5:
            sizes_to_use.append(np.nan)
            dists_to_use.append(np.nan)

        # Unpack sizes
        nn1_size.append(sizes_to_use[0])
        nn2_size.append(sizes_to_use[1])
        nn3_size.append(sizes_to_use[2])
        nn4_size.append(sizes_to_use[3])
        nn5_size.append(sizes_to_use[4])

        # Unpack distances
        nn1_dist.append(dists_to_use[0])
        nn2_dist.append(dists_to_use[1])
        nn3_dist.append(dists_to_use[2])
        nn4_dist.append(dists_to_use[3])
        nn5_dist.append(dists_to_use[4])

    # Assign everything back to the dataframe
    gdf["nn1_garden_size"] = nn1_size
    gdf["nn2_garden_size"] = nn2_size
    gdf["nn3_garden_size"] = nn3_size
    gdf["nn4_garden_size"] = nn4_size
    gdf["nn5_garden_size"] = nn5_size

    gdf["nn1_distance_m"] = nn1_dist
    gdf["nn2_distance_m"] = nn2_dist
    gdf["nn3_distance_m"] = nn3_dist
    gdf["nn4_distance_m"] = nn4_dist
    gdf["nn5_distance_m"] = nn5_dist

    return gdf


# %%
def _get_gdf_number_uprns_within_radius(gdf, radius_m=100, return_uprn_level=False):
    """
    Find the number of UPRNs within a given radius of building footprint centroids or UPRN point coordinates. If building footprint centroids, the UPRNs within the building are not counted.
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing building footprint polygons or UPRN point coordinates
        return_uprn_level (bool): Set True if gdf contains UPRN point coordinates, False if building footprints
    Returns:
        gpd.GeoDataFrame: number of UPRNs within buffer radius of each building centroid or UPRN point coordinate

    """

    # set the ID column and the "weight" of each point
    if return_uprn_level:
        unique_id_col = "UPRN"
        gdf["uprn_weight"] = 1
    else:
        # Each row represents a building.
        # The unique identifier is the ID column, and the 'UPRN' column holds the count.
        unique_id_col = "ID"
        gdf["uprn_weight"] = gdf["uprns_in_building"]

    # Create buffers around the centroids
    buffers = gpd.GeoDataFrame(
        {
            unique_id_col: gdf[unique_id_col],
            "geometry": gdf.geometry.centroid.buffer(radius_m),
        },
        crs=gdf.crs,
    )

    # Create the target points (buildings or uprn points)
    points = gpd.GeoDataFrame(
        {
            "target_id": gdf[unique_id_col],
            "uprn_weight": gdf["uprn_weight"],
            "geometry": gdf.geometry.centroid,
        },
        crs=gdf.crs,
    )

    # find target points inside buffer radius
    joined = gpd.sjoin(buffers, points, how="left", predicate="contains")

    # Group by the correct unique ID and sum the weights (count UPRNs within buffer radius)
    uprn_sums = joined.groupby(unique_id_col)["uprn_weight"].sum().reset_index()
    uprn_sums.rename(columns={"uprn_weight": "uprns_within_100m"}, inplace=True)

    # remove temporary weight column we added to the original GeoDataFrame
    gdf.drop(columns=["uprn_weight"], inplace=True)

    return uprn_sums


# %%
def _calculate_gdf_plot_ratio_proxy(buildings_gdf, buffer_radius=100):
    """
    Calculate the ratio of building footprint area contained within a buffer radius to the area of the buffer radius. E.g. a ratio of 1 means the building footprint takes up the whole buffer circle area, and a low area means the building footprint takes up little of the buffer radius
    Args:
        buildings_gdf (gpd.GeoDataFrame): building footprint polygons
        buffer_radius (int): distance (m) around building footprint centroid to calculate plot ratio for (default 100m)

    Returns:
        gpd.GeoDataFrame: building footprint polygons with plot ratio proxy in a new column
    """
    # calculate the area of the buffer circle
    buffer_area = np.pi * (buffer_radius**2)

    # Create the buffers
    buffers = buildings_gdf.geometry.centroid.buffer(buffer_radius)
    buffers_gdf = gpd.GeoDataFrame(geometry=buffers, index=buildings_gdf.index)

    # find buildings within buffer circles and keep the buffer geometry
    joined = gpd.sjoin(buffers_gdf, buildings_gdf, how="inner", predicate="intersects")

    left_geoms = joined.geometry  # These are the buffer circles

    # get building polygons that are within the buffer circles
    # the geometry will be of the full building (not just the bit inside the circle)
    # explicitly give them the exact same index as the 'joined' dataframe.
    right_geoms = gpd.GeoSeries(
        buildings_gdf.loc[joined["index_right"], "geometry"].values, index=joined.index
    )

    # now clip building geometries to just what is within the buffer circle areas
    exact_intersections = left_geoms.intersection(right_geoms)

    # Calculate the area of just those clipped pieces
    joined["clipped_area"] = exact_intersections.area

    # Group by the buffer's index and sum the clipped areas
    total_exact_area = joined.groupby(joined.index)["clipped_area"].sum()

    # Calculate the ratio and assign it back to the main dataframe
    buildings_gdf["plot_ratio_proxy"] = total_exact_area / buffer_area

    # Fill NaNs with 0 (in case a point had zero intersecting buildings)
    buildings_gdf["plot_ratio_proxy"] = buildings_gdf["plot_ratio_proxy"].fillna(0)

    return buildings_gdf


# %%
def _get_int_count_vertices(geom):
    """
    Counts the number of vertices in a polygon or mutipolygon
    Args:
        geom (shapely.Polygon | shapely.MultiPolygon): shapely geometry representing building footprint polygon

    Returns:
        int: count of vertices in the polygon or multi polygon
    """
    if geom.geom_type == "Polygon":
        # Count the coordinates forming the outer boundary
        return len(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        # Sum the coordinates for all pieces of the multi-polygon
        return sum(len(poly.exterior.coords) for poly in geom.geoms)
    return 0


# %%
# I have not made this work with the building level data yet but we aren't training on that anyway
def _compute_voronoi_area(gdf, grid_squares, boundary):
    """
    Calculates the area within voronoi polygons formed for each UPRN coordinate, with barrier features removed
    Args:
        gdf (gpd.GeoDataFrame): UPRN point coordinates
        grid_squares (str): grid squares to get the barrier features for
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary area to clip the voronoi polygons to

    Returns:
        gpd.GeoDataFrame: original gdf with column for voronoi cell (minus barrier features) area in m2
    """
    # Add an internal unique ID to each building
    id_col = "_internal_building_id"
    gdf[id_col] = np.arange(len(gdf))

    # Convert to a Multipoint collection for Voronoi
    coords = shapely.MultiPoint(gdf.geometry.tolist())

    print("Computing Voronoi diagram...")
    # Compute Voronoi polygons up to specified boundary, create one Voronoi cell per point
    voronoi_collection = shapely.voronoi_polygons(coords)

    # Convert to a geodataframe
    voronoi_gdf = gpd.GeoDataFrame(geometry=list(voronoi_collection.geoms), crs=gdf.crs)

    print(
        "Joining Voronois to original building footprints and dissolving per footprint..."
    )
    # Join the original building points with IDs to the Voronoi cells and dissolve to get one polygon per internal building ID
    voronoi_gdf = (
        voronoi_gdf.sjoin(gdf, how="inner", predicate="contains")
        .dissolve(by=id_col)
        .reset_index()
    ).clip(boundary["geometry"])

    voronoi_gdf = voronoi_gdf[[id_col, "geometry"]].reset_index(drop=True)

    polygon_barriers = cluster.load_transform_gdf_polygon_barriers(
        grid_squares=grid_squares
    )[["geometry"]].reset_index(drop=True)
    linestring_barriers = cluster.load_tranform_gdf_linestring_barriers(
        grid_squares=grid_squares
    )[["geometry"]].reset_index(drop=True)

    voronoi_gdf = (
        voronoi_gdf.overlay(polygon_barriers, how="difference")
        .overlay(linestring_barriers, how="difference")
        .explode()
    )

    voronoi_gdf["voronoi_area"] = voronoi_gdf.area

    area_summary = voronoi_gdf.groupby(id_col)["voronoi_area"].sum().reset_index()
    final_gdf = gdf.merge(area_summary, on=id_col, how="left")
    final_gdf["voronoi_area"] = final_gdf["voronoi_area"].fillna(0)

    final_gdf = final_gdf.drop(columns=[id_col])

    return final_gdf


# %%
aggregation_rules = {
    "spatial_signature_types": "first",
    "UPRN": "count",
    "max_contiguous_outdoor_space_area_m2": "first",
    "ATTACHMENT": "first",
}


def engineer_gdf_features(local_authorities, return_uprn_level=False):
    """
    Engineer set of features for model training
    Args:
        local_authorities (str | list[str]): Local Authority or Authorities to engineer features for
        return_uprn_level: whether to return the features at the building or UPRN level
    Returns:
        gpd.GeoDataFrame: features for model training
    """

    grid_squares = list(local_authority.get_list_la_grid_squares(local_authorities))

    # get building footprints for LA boundary
    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        local_authorities
    )
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=grid_squares
    )
    buildings_gdf = buildings_gdf.clip(boundary_gdf)

    # add building level features
    buildings_gdf["building_area_m2"] = buildings_gdf.area
    buildings_gdf["building_perimeter_m"] = buildings_gdf.length
    buildings_gdf = _calculate_gdf_plot_ratio_proxy(buildings_gdf)

    # get convex hull
    convex_hull_areas = buildings_gdf.geometry.convex_hull.area

    # Divide the actual building area by the convex hull area
    buildings_gdf["building_convexity"] = (
        buildings_gdf.geometry.area / convex_hull_areas
    )

    # Fill any 0-division errors etc with 1.0
    buildings_gdf["building_convexity"] = buildings_gdf["building_convexity"].fillna(
        1.0
    )

    # get number of points in a building polygon
    buildings_gdf["building_vertex_count"] = buildings_gdf.geometry.apply(
        _get_int_count_vertices
    )

    # load uprns with features
    uprns_df_list = []
    for la in local_authorities:
        la_slug = local_authority.make_str_slug(la)
        uprns_la = pd.read_parquet(
            f"s3://asf-local-heat-planning-tool/outputs/data/{la_slug}/{la_slug}_with_features.parquet"
        )
        uprns_df_list.append(uprns_la)

    uprns_df = pd.concat(uprns_df_list, ignore_index=True)
    uprns_df = uprns_df.dropna(subset=["ID"])

    # if returning uprn level dataframe
    if return_uprn_level:

        # add column of UPRNs per building footprint
        uprn_counts = (
            uprns_df.groupby("ID").size().reset_index(name="uprns_in_building")
        )
        uprns_df = uprns_df.merge(uprn_counts, on="ID", how="left")
        df_with_features = pd.merge(uprns_df, buildings_gdf, on="ID", how="left")

        gdf_with_features = gpd.GeoDataFrame(
            df_with_features,
            geometry=gpd.points_from_xy(
                df_with_features["X_COORDINATE"], df_with_features["Y_COORDINATE"]
            ),
        )

        # used for nearest neighbour based features
        unique_id_col = "UPRN"

    # if returning building level dataframe
    else:

        # uprns per building footprint and aggregate uprn features to building level
        uprn_counts = uprns_df.groupby("ID").agg(aggregation_rules).reset_index()
        uprn_counts.rename(columns={"UPRN": "uprns_in_building"}, inplace=True)

        gdf_with_features = pd.merge(uprn_counts, buildings_gdf, on="ID", how="left")

        gdf_with_features = gpd.GeoDataFrame(gdf_with_features, geometry="geometry")

        # used for nearest neighbour analysis
        unique_id_col = "ID"

    gdf_with_features["area_per_uprn"] = (
        gdf_with_features["building_area_m2"] / gdf_with_features["uprns_in_building"]
    )
    gdf_with_features["perimeter_to_area_ratio"] = (
        gdf_with_features["building_perimeter_m"]
        / gdf_with_features["building_area_m2"]
    )

    gdf_with_features = _compute_voronoi_area(
        gdf_with_features, grid_squares, boundary_gdf
    )

    # convert spatial signature types to be one column per spatial signature type, with 1 being True and 0 being False
    gdf_with_features["spatial_signature_types"] = (
        gdf_with_features["spatial_signature_types"]
        .astype(str)
        .str.replace(r"[\[\]\'\"]", "", regex=True)
    )
    gdf_with_features = pd.get_dummies(
        gdf_with_features,
        columns=["spatial_signature_types"],
        prefix="spatial_signature",
        dtype=int,
    )
    # as above, for attachment type
    gdf_with_features = pd.get_dummies(
        gdf_with_features, columns=["ATTACHMENT"], prefix="ATTACHMENT", dtype=int
    )
    gdf_with_features = _get_gdf_5nn_spatial_features(gdf_with_features, unique_id_col)

    uprn_sums = _get_gdf_number_uprns_within_radius(
        gdf=gdf_with_features, return_uprn_level=return_uprn_level
    )
    gdf_with_features = gdf_with_features.merge(uprn_sums, on=unique_id_col, how="left")

    return gdf_with_features


# %%
def prepare_test_train_datasets(gdf, feature_cols):
    """
    Split dataset into test and train sets, keeping buildings/UPRNs in the same area in the same set

    Args:
        gdf (gpd.GeoDataFrame): feature dataset
        feature_cols (list[str]): names of columns to use as features for training

    Returns:
        dict: features dataset split into test and train sets, with groups assigned based on spatial clustering
    """
    # TODO: think about these more. Are there lots of na values in dataset?
    gdf = gdf.fillna(
        {
            #'mean_5nn_garden_size': df['mean_5nn_garden_size'].median(),
            "uprns_within_100m": 0,
            "area_per_uprn": gdf["area_per_uprn"].median(),
        }
    )

    print("Splitting datasets...")
    # Buildings where we already know the garden size for training/ testing
    known_gdf = gdf[gdf["max_contiguous_outdoor_space_area_m2"].notna()].copy()

    # Prediction data: Buildings missing their garden size
    predict_gdf = gdf[gdf["max_contiguous_outdoor_space_area_m2"].isna()].copy()

    # group the known data into spatial clusters
    coords = np.column_stack(
        (known_gdf.geometry.centroid.x, known_gdf.geometry.centroid.y)
    )
    spatial_clusters = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(
        coords
    )
    known_gdf["spatial_cluster"] = spatial_clusters

    # get just the features we care about
    features = [col for col in known_gdf.columns if col in feature_cols]
    X = known_gdf[features]
    y = known_gdf["max_contiguous_outdoor_space_area_m2"]
    groups_known = known_gdf["spatial_cluster"]
    X_predict = predict_gdf[features]

    # split known data into test and train datasets using group shuffle split. This keeps data in the same spatial cluster within the same set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, spatial_clusters))
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    # give the training data an index based on their group
    groups_train = groups_known.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    # initialise a group k-fold split to be used in the training step
    gkf = GroupKFold(n_splits=4)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_predict": X_predict,
        "gkf": gkf,
        "groups_train": groups_train,
        "features": features,
    }


# %%
local_authorities = ["Plymouth"]

# %%
correlation_cols = [
    "max_contiguous_outdoor_space_area_m2",
    "uprns_in_building",
    "building_area_m2",
    "area_per_uprn",
    "spatial_signature_Accessible suburbia",
    "spatial_signature_Connected residential neighbourhoods",
    "spatial_signature_Countryside agriculture",
    "spatial_signature_Dense residential neighbourhoods",
    "spatial_signature_Dense urban neighbourhoods",
    "spatial_signature_Disconnected suburbia",
    "spatial_signature_Gridded residential quarters",
    "spatial_signature_Local urbanity",
    "spatial_signature_None",
    "spatial_signature_Open sprawl",
    "spatial_signature_Regional urbanity",
    "spatial_signature_Urban buffer",
    "spatial_signature_Warehouse/Park land",
    "spatial_signature_Wild countryside",
    "ATTACHMENT_Detached",
    "ATTACHMENT_End-Terrace",
    "ATTACHMENT_Flat",
    "ATTACHMENT_Mid-Terrace",
    "ATTACHMENT_Semi-Detached",
    "uprns_within_100m",
    "plot_ratio_proxy",
    "perimeter_to_area_ratio",
    "building_convexity",
    "building_vertex_count",
    "voronoi_area",
    "nn1_garden_size",
    "nn2_garden_size",
    "nn3_garden_size",
    "nn4_garden_size",
    "nn5_garden_size",
    "nn1_distance_m",
    "nn2_distance_m",
    "nn3_distance_m",
    "nn4_distance_m",
    "nn5_distance_m",
]

# %%
feature_cols = [
    "uprns_in_building",
    "building_area_m2",
    "area_per_uprn",
    "spatial_signature_Accessible suburbia",
    "spatial_signature_Connected residential neighbourhoods",
    "spatial_signature_Countryside agriculture",
    "spatial_signature_Dense residential neighbourhoods",
    "spatial_signature_Dense urban neighbourhoods",
    "spatial_signature_Disconnected suburbia",
    "spatial_signature_Gridded residential quarters",
    "spatial_signature_Local urbanity",
    "spatial_signature_None",
    "spatial_signature_Open sprawl",
    "spatial_signature_Regional urbanity",
    "spatial_signature_Urban buffer",
    "spatial_signature_Warehouse/Park land",
    "spatial_signature_Wild countryside",
    "ATTACHMENT_Detached",
    "ATTACHMENT_End-Terrace",
    "ATTACHMENT_Flat",
    "ATTACHMENT_Mid-Terrace",
    "ATTACHMENT_Semi-Detached",
    "uprns_within_100m",
    "plot_ratio_proxy",
    "perimeter_to_area_ratio",
    "building_convexity",
    "building_vertex_count",
    "voronoi_area",
    "nn1_garden_size",
    "nn2_garden_size",
    "nn3_garden_size",
    "nn4_garden_size",
    "nn5_garden_size",
    "nn1_distance_m",
    "nn2_distance_m",
    "nn3_distance_m",
    "nn4_distance_m",
    "nn5_distance_m",
]

# %% [markdown]
# ## Building Level

# %%
# df = engineer_gdf_features(grid_squares= grid_squares, local_authorities = local_authorities)

# %%
"""
features = [col for col in df.columns if col in correlation_cols]
corr_matrix = df[features].corr()
target_corr = corr_matrix[['max_contiguous_outdoor_space_area_m2']].drop('max_contiguous_outdoor_space_area_m2')
target_corr = target_corr.sort_values(by='max_contiguous_outdoor_space_area_m2', ascending=False)
# Plot the heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Feature Correlation with Garden Size: Building Level')
plt.tight_layout()
plt.show()
"""

# %%
# test_train_dict = prepare_test_train_datasets(df, feature_cols=feature_cols)

# %%
# lin_reg = LinearRegression()
# lin_reg.fit(test_train_dict['X_train'], test_train_dict['y_train'])

# %%
# lin_predictions = lin_reg.predict(test_train_dict['X_test'])

# %%
"""
rf = RandomForestRegressor()

param_distributions = {
    'n_estimators': [100, 250, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 1.0]
}
"""

# %%
"""
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=15,
    cv=test_train_dict['gkf'],                  # Uses spatial GroupKFold
    n_jobs=-1,
    scoring='neg_mean_absolute_error'
)
"""

# %%
# rf_random.fit(test_train_dict['X_train'], test_train_dict['y_train'], groups=test_train_dict['groups_train'])

# %%
"""
best_model = rf_random.best_estimator_
print(f"\nBest Parameters: {rf_random.best_params_}")
test_predictions = best_model.predict(test_train_dict['X_test'])
"""

# %%
"""
bdt = GradientBoostingRegressor()

bdt_param_distributions = {
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.8, 0.9, 1.0]
}
"""

# %%
"""
bdt_random = RandomizedSearchCV(
    estimator=bdt,
    param_distributions=bdt_param_distributions,
    n_iter=15,
    cv=test_train_dict['gkf'],
    n_jobs=-1,
    scoring='neg_mean_absolute_error'
)
"""

# %%
# bdt_random.fit(test_train_dict['X_train'], test_train_dict['y_train'], groups=test_train_dict['groups_train'])

# %%
# bdt_best = bdt_random.best_estimator_
# bdt_predictions = bdt_best.predict(test_train_dict['X_test'])

# %%
"""
print("\n--- Side-by-Side Model Comparison ---")

print("\n--- Linear Regression Performance: ---")
print(f"R-squared: {r2_score(test_train_dict['y_test'], lin_predictions):.3f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(test_train_dict['y_test'], lin_predictions):.2f}")
print(f"Median Absolute Error (MedAE): {median_absolute_error(test_train_dict['y_test'], lin_predictions):.2f}")

print("\n--- Random Forest: ---")
print(f"   R-squared: {r2_score(test_train_dict['y_test'], test_predictions):.3f}")
print(f"   MAE:       {mean_absolute_error(test_train_dict['y_test'], test_predictions):.2f} m2")
print(f"   MedAE:     {median_absolute_error(test_train_dict['y_test'], test_predictions):.2f} m2")

print("\n--- Boosted Decision Tree ---:")
print(f"   R-squared: {r2_score(test_train_dict['y_test'], bdt_predictions):.3f}")
print(f"   MAE:       {mean_absolute_error(test_train_dict['y_test'], bdt_predictions):.2f} m2")
print(f"   MedAE:     {median_absolute_error(test_train_dict['y_test'], bdt_predictions):.2f} m2")
"""

# %%
"""
# Extract feature importances from the best BDT model
importances = bdt_best.feature_importances_

# Create a DataFrame and sort it
importance_df = pd.DataFrame({'Feature': test_train_dict["features"], 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the Top 15 Features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
plt.title('Top 15 Most Important Predictors for Garden Size (BDT)')
plt.xlabel('Relative Importance (Adds up to 1.0)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
"""

# %% [markdown]
# ## UPRN level

# %%
uprn_df = engineer_gdf_features(
    local_authorities=local_authorities, return_uprn_level=True
)

# %%
features_uprn = [col for col in uprn_df.columns if col in correlation_cols]
corr_matrix = uprn_df[features_uprn].corr()
target_corr = corr_matrix[["max_contiguous_outdoor_space_area_m2"]].drop(
    "max_contiguous_outdoor_space_area_m2"
)
target_corr = target_corr.sort_values(
    by="max_contiguous_outdoor_space_area_m2", ascending=False
)
# Plot the heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(target_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Feature Correlation with Garden Size: UPRN Level")
plt.tight_layout()
plt.show()

# %%
uprn_test_train_dict = prepare_test_train_datasets(uprn_df, feature_cols=feature_cols)

# %%
uprn_test_train_dict["y_train"]

# %%
y_train_clipped = np.clip(uprn_test_train_dict["y_train"], a_min=None, a_max=300.0)

# %%
rf_uprn = RandomForestRegressor()

param_distributions = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [15, 25, 35, 50],
    "min_samples_split": [10, 20, 30],
    "min_samples_leaf": [5, 10, 15, 20],
    "max_features": ["sqrt", 0.33, 0.5],
}

# %%
rf_random_uprn = RandomizedSearchCV(
    estimator=rf_uprn,
    param_distributions=param_distributions,
    n_iter=15,
    cv=uprn_test_train_dict[
        "gkf"
    ],  # telling the rf to use Group K fold when splitting the training set into cross-validation sets
    n_jobs=-1,
    scoring="neg_mean_absolute_error",  # I tried a couple different metrics at this step and the results didn't change too much
)

# %%
rf_random_uprn.fit(
    uprn_test_train_dict["X_train"],
    y_train_clipped,
    groups=uprn_test_train_dict["groups_train"],
)  # here we tell the rf to use the spatial group index for the group K fold splitting

# %%
best_model_uprn = rf_random_uprn.best_estimator_
print(f"\nBest Parameters: {rf_random_uprn.best_params_}")

test_predictions_uprn = best_model_uprn.predict(uprn_test_train_dict["X_test"])

# %%
bdt_uprn = GradientBoostingRegressor()

bdt_param_distributions = {
    "n_estimators": [100, 250, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.8, 0.9, 1.0],
}

# %%
print("Training and optimizing Boosted Decision Tree (this may take a minute)...")
bdt_random_uprn = RandomizedSearchCV(
    estimator=bdt_uprn,
    param_distributions=bdt_param_distributions,
    n_iter=15,
    cv=uprn_test_train_dict["gkf"],
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
)

# %%
bdt_random_uprn.fit(
    uprn_test_train_dict["X_train"],
    y_train_clipped,
    groups=uprn_test_train_dict["groups_train"],
)

# %%
bdt_best_uprn = bdt_random_uprn.best_estimator_
bdt_predictions_uprn = bdt_best_uprn.predict(uprn_test_train_dict["X_test"])

# %%
print(f"\nBest Parameters: {bdt_random_uprn.best_params_}")


# %%
def calculate_mdape(y_true, y_pred, min_denominator=1.0):
    """
    Calculates the Median Absolute Percentage Error safely.
    min_denominator prevents division-by-zero errors for tiny/zero targets.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the raw absolute error for every row
    abs_errors = np.abs(y_true - y_pred)

    # Create a safe denominator (treats any true value smaller than 1.0 as 1.0). Otherwise this metric can blow up
    safe_denominator = np.maximum(y_true, min_denominator)

    # Calculate the percentage error for every row
    percentage_errors = abs_errors / safe_denominator

    # Return the median value of all those percentages
    return np.median(percentage_errors)


# %%
# clip true values over 300m2
y_test_clipped = np.clip(uprn_test_train_dict["y_test"], a_min=None, a_max=300.0)

# clip predicted values < 0 to 0m2 (this doesn't happen very often)
test_predictions_uprn = np.clip(test_predictions_uprn, a_min=0, a_max=None)
bdt_predictions_uprn = np.clip(bdt_predictions_uprn, a_min=0, a_max=None)

# %%
print("\n--- Side-by-Side Model Comparison ---")

print("\n--- Random Forest: ---")
print(f"   R-squared: {r2_score(y_test_clipped, test_predictions_uprn):.3f}")
print(
    f"   MAE:       {mean_absolute_error(y_test_clipped, test_predictions_uprn):.2f} m2"
)
print(
    f"   MedAE:     {median_absolute_error(y_test_clipped, test_predictions_uprn):.2f} m2"
)
print(f"   MSE:       {mean_squared_error(y_test_clipped, test_predictions_uprn):.2f}")
print(
    f"   MAPE:      {mean_absolute_percentage_error(y_test_clipped, test_predictions_uprn):.2f}"
)  # this one blows up I think because of really small gardens.
print(
    f"   RMSLE:     {root_mean_squared_log_error(y_test_clipped, test_predictions_uprn):.2f}"
)
print(
    f"   MedAPE:    {100*calculate_mdape(y_test_clipped, test_predictions_uprn):.2f}%"
)

print("\n--- Boosted Decision Tree ---:")
print(f"   R-squared: {r2_score(y_test_clipped, bdt_predictions_uprn):.3f}")
print(
    f"   MAE:       {mean_absolute_error(y_test_clipped, bdt_predictions_uprn):.2f} m2"
)
print(
    f"   MedAE:     {median_absolute_error(y_test_clipped, bdt_predictions_uprn):.2f} m2"
)
print(f"   MSE:       {mean_squared_error(y_test_clipped, bdt_predictions_uprn):.2f}")
print(
    f"   MAPE:      {mean_absolute_percentage_error(y_test_clipped, bdt_predictions_uprn):.2f}"
)
print(
    f"   RMSLE:     {root_mean_squared_log_error(y_test_clipped, bdt_predictions_uprn):.2f}"
)
print(f"   MedAPE:    {100*calculate_mdape(y_test_clipped, bdt_predictions_uprn):.2f}%")

# %%
# Extract feature importances from the best BDT model
importances_uprn = bdt_best_uprn.feature_importances_

# Create a DataFrame and sort it
importance_df_uprn = pd.DataFrame(
    {"Feature": uprn_test_train_dict["features"], "Importance": importances_uprn}
)
importance_df_uprn = importance_df_uprn.sort_values(by="Importance", ascending=False)

# Plot the Top 15 Features
plt.figure(figsize=(10, 8))
sns.barplot(
    x="Importance", y="Feature", data=importance_df_uprn.head(15), palette="viridis"
)
plt.title("Top 15 Most Important Predictors for Garden Size (BDT)")
plt.xlabel("Relative Importance (Adds up to 1.0)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# %%
