# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# %%
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata
from asf_heat_pump_suitability.pipeline.transform import local_authority


# %%
def _get_int_garden_area_nearest_neighbours(gdf, unique_id_col):

    all_centroids = np.column_stack((gdf.geometry.centroid.x, gdf.geometry.centroid.y))
    all_ids = gdf[unique_id_col].values

    known_gardens = gdf.dropna(subset=["max_contiguous_outdoor_space_area_m2"])
    known_centroids = np.column_stack(
        (known_gardens.geometry.centroid.x, known_gardens.geometry.centroid.y)
    )
    known_sizes = known_gardens["max_contiguous_outdoor_space_area_m2"].values
    known_ids = known_gardens[unique_id_col].values

    # We query 10 neighbours to ensure we have 5 left after dropping the self-match
    nn = NearestNeighbors(n_neighbors=10, algorithm="kd_tree")
    nn.fit(known_centroids)

    distances, indices = nn.kneighbors(all_centroids)

    mean_5nn = []

    for i in range(len(all_centroids)):
        target_id = all_ids[i]
        row_indices = indices[i]

        # Fetch the actual unique IDs of the matched neighbours
        neighbor_ids = known_ids[row_indices]

        # Keep the neighbour only if its ID does not match the target's ID.
        valid_mask = neighbor_ids != target_id

        # Apply the mask to get the sizes of only the valid neighbours
        valid_sizes = known_sizes[row_indices[valid_mask]]

        # Take the mean of the first 5 valid neighbours
        if len(valid_sizes) >= 5:
            mean_5nn.append(np.mean(valid_sizes[:5]))
        elif len(valid_sizes) > 0:
            mean_5nn.append(np.mean(valid_sizes))
        else:
            mean_5nn.append(np.nan)

    return mean_5nn


# %%
def _get_int_number_uprns_within_radius(gdf, return_uprn_level=False):
    radius_m = 100  # 100 meters

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
def _calculate_plot_ratio_proxy(buildings_gdf, buffer_radius=100):
    # calculate the area of the buffer circle
    buffer_area = 3.14159 * (buffer_radius**2)

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
def _calculate_nearest_building_centroid(buildings_gdf):
    # get centroid coordinates
    coords = np.column_stack(
        (buildings_gdf.geometry.centroid.x, buildings_gdf.geometry.centroid.y)
    )

    # find nearest neighbours of the centroids
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(coords)

    # since we are applying this to the same dataset (unlike in the 5 nearest neighbour garden size) don't need to exclude the point itself
    distances, indices = nn.kneighbors()

    # Extract the distance
    buildings_gdf["dist_to_nearest_building_centroid_m"] = distances[:, 0]

    return buildings_gdf


# %%
def _count_vertices(geom):
    if geom.geom_type == "Polygon":
        # Count the coordinates forming the outer boundary
        return len(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        # Sum the coordinates for all pieces of the multi-polygon
        return sum(len(poly.exterior.coords) for poly in geom.geoms)
    return 0


# %%
aggregation_rules = {
    "spatial_signature_types": "first",
    "UPRN": "count",
    "max_contiguous_outdoor_space_area_m2": "first",
    "ATTACHMENT": "first",
}


def engineer_gdf_features(grid_squares, local_authorities, return_uprn_level=False):

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
    buildings_gdf = _calculate_plot_ratio_proxy(buildings_gdf)
    buildings_gdf = _calculate_nearest_building_centroid(buildings_gdf)

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
        _count_vertices
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
    mean_neighbour_garden_size = _get_int_garden_area_nearest_neighbours(
        gdf_with_features, unique_id_col=unique_id_col
    )
    gdf_with_features["mean_5nn_garden_size"] = mean_neighbour_garden_size

    uprn_sums = _get_int_number_uprns_within_radius(
        gdf=gdf_with_features, return_uprn_level=return_uprn_level
    )
    gdf_with_features = gdf_with_features.merge(uprn_sums, on=unique_id_col, how="left")

    return gdf_with_features


# %%
def prepare_test_train_datasets(df, feature_cols):
    df = df.fillna(
        {
            "mean_5nn_garden_size": df["mean_5nn_garden_size"].median(),
            "uprns_within_100m": 0,
            "area_per_uprn": df["area_per_uprn"].median(),
        }
    )
    print("Splitting datasets...")
    # Training data: Buildings where we already know the garden size
    known_df = df[df["max_contiguous_outdoor_space_area_m2"].notna()].copy()

    # Prediction data: Buildings missing their garden size
    predict_df = df[df["max_contiguous_outdoor_space_area_m2"].isna()].copy()

    coords = np.column_stack(
        (known_df.geometry.centroid.x, known_df.geometry.centroid.y)
    )
    spatial_clusters = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(
        coords
    )
    known_df["spatial_cluster"] = spatial_clusters

    features = [col for col in known_df.columns if col in feature_cols]
    X_known = known_df[features]
    y_known = known_df["max_contiguous_outdoor_space_area_m2"]
    groups_known = known_df["spatial_cluster"]
    X_predict = predict_df[features]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_known, y_known, spatial_clusters))
    X_train = X_known.iloc[train_idx]
    y_train = y_known.iloc[train_idx]
    groups_train = groups_known.iloc[train_idx]
    X_test = X_known.iloc[test_idx]
    y_test = y_known.iloc[test_idx]
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
grid_squares = list(local_authority.get_list_la_grid_squares(local_authorities))

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
    "mean_5nn_garden_size",
    "uprns_within_100m",
    "plot_ratio_proxy",
    "perimeter_to_area_ratio",
    "dist_to_nearest_building_centroid_m",
    "building_convexity",
    "building_vertex_count",
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
    "mean_5nn_garden_size",
    "uprns_within_100m",
    "plot_ratio_proxy",
    "perimeter_to_area_ratio",
    "dist_to_nearest_building_centroid_m",
    "building_convexity",
    "building_vertex_count",
]

# %% [markdown]
# ## Building Level

# %%
df = engineer_gdf_features(
    grid_squares=grid_squares, local_authorities=local_authorities
)

# %%
features = [col for col in df.columns if col in correlation_cols]
corr_matrix = df[features].corr()
target_corr = corr_matrix[["max_contiguous_outdoor_space_area_m2"]].drop(
    "max_contiguous_outdoor_space_area_m2"
)
target_corr = target_corr.sort_values(
    by="max_contiguous_outdoor_space_area_m2", ascending=False
)
# Plot the heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(target_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Feature Correlation with Garden Size: Building Level")
plt.tight_layout()
plt.show()

# %%
test_train_dict = prepare_test_train_datasets(df, feature_cols=feature_cols)

# %%
lin_reg = LinearRegression()
lin_reg.fit(test_train_dict["X_train"], test_train_dict["y_train"])

# %%
lin_predictions = lin_reg.predict(test_train_dict["X_test"])

# %%
rf = RandomForestRegressor()

param_distributions = {
    "n_estimators": [100, 250, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 1.0],
}

# %%
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=15,
    cv=test_train_dict["gkf"],  # Uses spatial GroupKFold
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
)

# %%
rf_random.fit(
    test_train_dict["X_train"],
    test_train_dict["y_train"],
    groups=test_train_dict["groups_train"],
)

# %%
best_model = rf_random.best_estimator_
print(f"\nBest Parameters: {rf_random.best_params_}")
test_predictions = best_model.predict(test_train_dict["X_test"])

# %%
bdt = GradientBoostingRegressor()

bdt_param_distributions = {
    "n_estimators": [100, 250, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.8, 0.9, 1.0],
}

# %%
bdt_random = RandomizedSearchCV(
    estimator=bdt,
    param_distributions=bdt_param_distributions,
    n_iter=15,
    cv=test_train_dict["gkf"],
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
)

# %%
bdt_random.fit(
    test_train_dict["X_train"],
    test_train_dict["y_train"],
    groups=test_train_dict["groups_train"],
)

# %%
bdt_best = bdt_random.best_estimator_
bdt_predictions = bdt_best.predict(test_train_dict["X_test"])

# %%

print("\n--- Side-by-Side Model Comparison ---")

print("\n--- Linear Regression Performance: ---")
print(f"R-squared: {r2_score(test_train_dict['y_test'], lin_predictions):.3f}")
print(
    f"Mean Absolute Error (MAE): {mean_absolute_error(test_train_dict['y_test'], lin_predictions):.2f}"
)
print(
    f"Median Absolute Error (MedAE): {median_absolute_error(test_train_dict['y_test'], lin_predictions):.2f}"
)

print("\n--- Random Forest: ---")
print(f"   R-squared: {r2_score(test_train_dict['y_test'], test_predictions):.3f}")
print(
    f"   MAE:       {mean_absolute_error(test_train_dict['y_test'], test_predictions):.2f} m2"
)
print(
    f"   MedAE:     {median_absolute_error(test_train_dict['y_test'], test_predictions):.2f} m2"
)

print("\n--- Boosted Decision Tree ---:")
print(f"   R-squared: {r2_score(test_train_dict['y_test'], bdt_predictions):.3f}")
print(
    f"   MAE:       {mean_absolute_error(test_train_dict['y_test'], bdt_predictions):.2f} m2"
)
print(
    f"   MedAE:     {median_absolute_error(test_train_dict['y_test'], bdt_predictions):.2f} m2"
)

# %%
# Extract feature importances from the best BDT model
importances = bdt_best.feature_importances_

# Create a DataFrame and sort it
importance_df = pd.DataFrame(
    {"Feature": test_train_dict["features"], "Importance": importances}
)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot the Top 15 Features
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), palette="viridis")
plt.title("Top 15 Most Important Predictors for Garden Size (BDT)")
plt.xlabel("Relative Importance (Adds up to 1.0)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## UPRN level

# %%
uprn_df = engineer_gdf_features(
    grid_squares=grid_squares,
    local_authorities=local_authorities,
    return_uprn_level=True,
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
rf_uprn = RandomForestRegressor()

param_distributions = {
    "n_estimators": [100, 250, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 1.0],
}

# %%
rf_random_uprn = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=15,
    cv=uprn_test_train_dict["gkf"],
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
)

# %%
rf_random_uprn.fit(
    uprn_test_train_dict["X_train"],
    uprn_test_train_dict["y_train"],
    groups=uprn_test_train_dict["groups_train"],
)

# %%
best_model_uprn = rf_random_uprn.best_estimator_
print(f"\nBest Parameters: {rf_random_uprn.best_params_}")

# To get a realistic performance metric, we predict on the training data using cross-val
# (In a real strictly rigorous pipeline, you'd hold out a completely separate spatial test set)
test_predictions_uprn = best_model_uprn.predict(uprn_test_train_dict["X_test"])

# %%
bdt_uprn = GradientBoostingRegressor()

# BDTs have a 'learning_rate' parameter which is crucial to tune
bdt_param_distributions = {
    "n_estimators": [100, 250, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],  # Trees are usually shallower in boosting
    "subsample": [0.8, 0.9, 1.0],  # Helps prevent overfitting
}

# %%
print("Training and optimizing Boosted Decision Tree (this may take a minute)...")
bdt_random_uprn = RandomizedSearchCV(
    estimator=bdt_uprn,
    param_distributions=bdt_param_distributions,
    n_iter=15,
    cv=uprn_test_train_dict["gkf"],  # Reusing your exact same spatial GroupKFold
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
)

# %%
bdt_random_uprn.fit(
    uprn_test_train_dict["X_train"],
    uprn_test_train_dict["y_train"],
    groups=uprn_test_train_dict["groups_train"],
)

# %%
bdt_best_uprn = bdt_random_uprn.best_estimator_
bdt_predictions_uprn = bdt_best_uprn.predict(uprn_test_train_dict["X_test"])

# %%
print("\n--- Side-by-Side Model Comparison ---")

print("\n--- Random Forest: ---")
print(
    f"   R-squared: {r2_score(uprn_test_train_dict['y_test'], test_predictions_uprn):.3f}"
)
print(
    f"   MAE:       {mean_absolute_error(uprn_test_train_dict['y_test'], test_predictions_uprn):.2f} m2"
)
print(
    f"   MedAPE:     {median_absolute_error(uprn_test_train_dict['y_test'], test_predictions_uprn):.2f} m2"
)

print("\n--- Boosted Decision Tree ---:")
print(
    f"   R-squared: {r2_score(uprn_test_train_dict['y_test'], bdt_predictions_uprn):.3f}"
)
print(
    f"   MAE:       {mean_absolute_error(uprn_test_train_dict['y_test'], bdt_predictions_uprn):.2f} m2"
)
print(
    f"   MedAE:     {median_absolute_error(uprn_test_train_dict['y_test'], bdt_predictions_uprn):.2f} m2"
)

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
