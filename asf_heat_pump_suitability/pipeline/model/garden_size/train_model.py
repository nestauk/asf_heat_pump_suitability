import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    root_mean_squared_log_error,
    classification_report,
    confusion_matrix,
)

CORRELATION_COLS = [
    "max_contiguous_outdoor_space_area_m2",
    "n_uprns_in_building",
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

FEATURE_COLS = [
    "n_uprns_in_building",
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


def prepare_test_train_datasets(gdf, feature_cols):
    """
    Split dataset into test and train sets, keeping UPRNs in the same area in the same set.

    Args:
        gdf (gpd.GeoDataFrame): Feature dataset containing UPRN point coordinates and predictors.
        feature_cols (list[str]): Names of columns to use as features for training.

    Returns:
        dict: A dictionary containing the X/y splits, prediction set, k-fold splitter,
              training groups, and feature names.
    """
    # TODO: think about these more. Are there lots of na values in dataset?
    #  handle all NaNs in the features before training
    nn_size_cols = [
        "nn1_garden_size",
        "nn2_garden_size",
        "nn3_garden_size",
        "nn4_garden_size",
        "nn5_garden_size",
    ]
    median_garden_size = gdf["max_contiguous_outdoor_space_area_m2"].median()

    fill_dict = {
        "uprns_within_100m": 0,
        "area_per_uprn": gdf["area_per_uprn"].median(),
        "plot_ratio_proxy": 0,
        "building_convexity": 1.0,
        "nn1_distance_m": 9999,  # Massive distance penalty for missing neighbors
        "nn2_distance_m": 9999,
        "nn3_distance_m": 9999,
        "nn4_distance_m": 9999,
        "nn5_distance_m": 9999,
    }

    # Add the median garden size to the fill dictionary for all 5 neighbor columns
    for col in nn_size_cols:
        fill_dict[col] = median_garden_size

    gdf = gdf.fillna(fill_dict)

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


def plot_and_save_correlation_matrix(
    uprn_df, correlation_cols, filename="correlation_matrix.png", save=False
):
    """
    Generates and optionally saves a Seaborn heatmap showing feature correlation with garden size.

    Args:
        uprn_df (pd.DataFrame): Dataframe containing features and the target variable.
        correlation_cols (list[str]): List of columns to include in the correlation matrix.
        filename (str, optional): Name of the file to save the plot as. Defaults to 'correlation_matrix.png'.
        save (bool, optional): If True, saves the plot. Otherwise, displays it. Defaults to False.

    Returns:
        None
    """

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
    if save:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved plot as {filename}")
    else:
        plt.show()


def train_random_forest_regressor(uprn_test_train_dict, y_train, save=False):
    """
    Trains and hyperparameter-tunes a Random Forest Regressor using GroupKFold cross-validation.

    Args:
        uprn_test_train_dict (dict): Dictionary output from prepare_test_train_datasets containing X_train, groups, and gkf.
        y_train (np.ndarray | pd.Series): The target variable array (garden sizes) corresponding to X_train.
        save (bool, optional): Flag to indicate if model should be saved. Defaults to False.

    Returns:
        RandomForestRegressor: The best fit scikit-learn estimator found during the grid search.
    """
    rf_uprn = RandomForestRegressor()

    param_distributions = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [15, 25, 35, 50],
        "min_samples_split": [10, 20, 30],
        "min_samples_leaf": [5, 10, 15, 20],
        "max_features": ["sqrt", 0.33, 0.5],
    }

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

    rf_random_uprn.fit(
        uprn_test_train_dict["X_train"],
        y_train,
        groups=uprn_test_train_dict["groups_train"],
    )  # here we tell the rf to use the spatial group index for the group K fold splitting

    best_model_uprn = rf_random_uprn.best_estimator_
    print(f"\nBest Parameters: {rf_random_uprn.best_params_}")

    return best_model_uprn


def train_boosted_decision_tree(uprn_test_train_dict, y_train, save=False):
    """
    Trains and hyperparameter-tunes a Gradient Boosting Regressor using GroupKFold cross-validation.

    Args:
        uprn_test_train_dict (dict): Dictionary output from prepare_test_train_datasets containing X_train, groups, and gkf.
        y_train (np.ndarray | pd.Series): The target variable array (garden sizes) corresponding to X_train.
        save (bool, optional): Flag to indicate if model should be saved. Defaults to False.

    Returns:
        GradientBoostingRegressor: The best fit scikit-learn estimator found during the grid search.
    """

    bdt_uprn = GradientBoostingRegressor()

    bdt_param_distributions = {
        "n_estimators": [100, 250, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.8, 0.9, 1.0],
    }

    bdt_random_uprn = RandomizedSearchCV(
        estimator=bdt_uprn,
        param_distributions=bdt_param_distributions,
        n_iter=15,
        cv=uprn_test_train_dict["gkf"],
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    bdt_random_uprn.fit(
        uprn_test_train_dict["X_train"],
        y_train,
        groups=uprn_test_train_dict["groups_train"],
    )

    bdt_best_uprn = bdt_random_uprn.best_estimator_
    print(f"\nBest Parameters: {bdt_random_uprn.best_params_}")

    return bdt_best_uprn


def calculate_mdape(y_true, y_pred, min_denominator=1.0):
    """
    Calculates the Median Absolute Percentage Error.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.
        min_denominator (float, optional): Baseline denominator to prevent division-by-zero errors for tiny/zero targets. Defaults to 1.0.

    Returns:
        float: The median absolute percentage error.
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


def plot_bdt_feature_importance(
    bdt, uprn_test_train_dict, filename="bdt_feature_importance.png", save=False
):
    """
    Extracts and plots the top 15 most important features from a trained Gradient Boosting model.

    Args:
        bdt (GradientBoostingRegressor): The trained model to extract importances from.
        uprn_test_train_dict (dict): Dictionary containing the list of feature names.
        filename (str, optional): Name of the file to save the plot as. Defaults to 'bdt_feature_importance.png'.
        save (bool, optional): If True, saves the plot. Otherwise, displays it. Defaults to False.

    Returns:
        None
    """

    # Extract feature importances from the best BDT model
    importances_uprn = bdt.feature_importances_

    # Create a DataFrame and sort it
    importance_df_uprn = pd.DataFrame(
        {"Feature": uprn_test_train_dict["features"], "Importance": importances_uprn}
    )
    importance_df_uprn = importance_df_uprn.sort_values(
        by="Importance", ascending=False
    )

    # Plot the Top 15 Features
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="Importance", y="Feature", data=importance_df_uprn.head(15), palette="viridis"
    )
    plt.title("Top 15 Most Important Predictors for Garden Size (BDT)")
    plt.xlabel("Relative Importance (Adds up to 1.0)")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved plot as {filename}")
    else:
        plt.show()


def plot_actual_vs_predicted_garden_size(predicted, actual, filename, save=False):
    """
    Generates a scatterplot comparing actual true garden sizes vs. model predictions, capped at 70m2.

    Args:
        predicted (array-like): Model predicted values.
        actual (array-like): Ground truth target values.
        filename (str): Name of the file to save the plot as.
        save (bool, optional): If True, saves the plot. Otherwise, displays it. Defaults to False.

    Returns:
        None
    """

    plt.figure(figsize=(8, 8))

    plt.scatter(
        x=predicted,  # predicted garden sizes
        y=actual,  # actual garden sizes
        alpha=0.5,
        edgecolor="white",
        color="#1f77b4",
    )

    # reference line
    plt.plot(
        [0, 70],
        [0, 70],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    # focussing on the smaller gardens where we care about this prediction more
    plt.xlim(0, 70)
    plt.ylim(0, 70)

    plt.title("Actual vs Predicted Garden Size (0-70m²)", fontsize=14, pad=15)
    plt.xlabel("Model's Predicted Size (m²)", fontsize=12)
    plt.ylabel("True Garden Size (m²)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    if save:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved plot as {filename}")
    else:
        plt.show()


def plot_buckets_confusion_matrix(predicted, actual, filename, save=False):
    """
    Bins continuous garden sizes into categorical buckets (<30, 30-70, >70) and generates a confusion matrix.

    Args:
        predicted (array-like): Model predicted values.
        actual (array-like): Ground truth target values.
        filename (str): Name of the file to save the plot as.
        save (bool, optional): If True, saves the plot. Otherwise, displays it. Defaults to False.

    Returns:
        None
    """

    bins = [-np.inf, 30, 70, np.inf]
    bucket_names = ["< 30m²", "30-70m²", "> 70m²"]

    # Convert the continuous target arrays into categorical buckets
    y_test_binned = pd.cut(actual, bins=bins, labels=bucket_names)
    y_pred_binned = pd.cut(predicted, bins=bins, labels=bucket_names)

    print("--- Garden Size Classification Report ---")
    print(
        classification_report(y_test_binned, y_pred_binned, target_names=bucket_names)
    )

    # Generate and plot the Confusion Matrix
    cm = confusion_matrix(
        y_test_binned, y_pred_binned, labels=bucket_names, normalize="true"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=bucket_names,
        yticklabels=bucket_names,
    )

    plt.title("Garden Size Bucket Predictions: True vs Predicted", pad=20, fontsize=14)
    plt.ylabel("TRUE Real-World Size", fontsize=12, fontweight="bold")
    plt.xlabel("MODEL Predicted Size", fontsize=12, fontweight="bold")

    if save:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Saved plot as {filename}")
    else:
        plt.show()


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire.",
        type=str,
        nargs="+",
        default="GB",
        required=False,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )


if __name__ == "__main__":
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.pipeline.model.garden_size import feature_engineering
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()

    local_authorities = [la.lower() for la in args.local_authorities]

    # add features to domestic uprn dataset
    uprn_df = feature_engineering.engineer_gdf_features(
        local_authorities=local_authorities
    )

    # extract test and train data
    uprn_test_train_dict = prepare_test_train_datasets(
        uprn_df, feature_cols=FEATURE_COLS
    )

    # clip large gardens so they don't confuse the model
    y_train_clipped = np.clip(uprn_test_train_dict["y_train"], a_min=None, a_max=300.0)

    # random forest training
    print("Training and optimizing Random Forest Regressor ...")
    rf_model = train_random_forest_regressor(
        uprn_test_train_dict=uprn_test_train_dict, y_train=y_train_clipped
    )

    # bdt training
    print("Training and optimizing Boosted Decision Tree ...")
    bdt_model = train_boosted_decision_tree(
        uprn_test_train_dict=uprn_test_train_dict, y_train=y_train_clipped
    )

    # make predictions on hold out test data
    rf_predictions = rf_model.predict(uprn_test_train_dict["X_test"])
    bdt_predictions = bdt_model.predict(uprn_test_train_dict["X_test"])

    # clip true values of hold out test data over 300m2
    y_test_clipped = np.clip(uprn_test_train_dict["y_test"], a_min=None, a_max=300.0)

    # clip predicted values < 0 to 0m2 (this doesn't happen very often)
    rf_predictions = np.clip(rf_predictions, a_min=0, a_max=None)
    bdt_predictions = np.clip(bdt_predictions, a_min=0, a_max=None)

    print("\n--- Side-by-Side Model Comparison ---")

    print("\n--- Random Forest: ---")
    print(f"   R-squared: {r2_score(y_test_clipped, rf_predictions):.3f}")
    print(f"   MAE:       {mean_absolute_error(y_test_clipped, rf_predictions):.2f} m2")
    print(
        f"   MedAE:     {median_absolute_error(y_test_clipped, rf_predictions):.2f} m2"
    )
    print(f"   MSE:       {mean_squared_error(y_test_clipped, rf_predictions):.2f}")
    print(
        f"   MAPE:      {mean_absolute_percentage_error(y_test_clipped, rf_predictions):.2f}"
    )  # this one blows up I think because of really small gardens.
    print(
        f"   RMSLE:     {root_mean_squared_log_error(y_test_clipped, rf_predictions):.2f}"
    )
    print(f"   MedAPE:    {100*calculate_mdape(y_test_clipped, rf_predictions):.2f}%")

    print("\n--- Boosted Decision Tree ---:")
    print(f"   R-squared: {r2_score(y_test_clipped, bdt_predictions):.3f}")
    print(
        f"   MAE:       {mean_absolute_error(y_test_clipped, bdt_predictions):.2f} m2"
    )
    print(
        f"   MedAE:     {median_absolute_error(y_test_clipped, bdt_predictions):.2f} m2"
    )
    print(f"   MSE:       {mean_squared_error(y_test_clipped, bdt_predictions):.2f}")
    print(
        f"   MAPE:      {mean_absolute_percentage_error(y_test_clipped, bdt_predictions):.2f}"
    )
    print(
        f"   RMSLE:     {root_mean_squared_log_error(y_test_clipped, bdt_predictions):.2f}"
    )
    print(f"   MedAPE:    {100*calculate_mdape(y_test_clipped, bdt_predictions):.2f}%")

    plot_bdt_feature_importance(
        bdt=bdt_model, uprn_test_train_dict=uprn_test_train_dict, save=args.save
    )

    plot_actual_vs_predicted_garden_size(
        predicted=rf_predictions,
        actual=y_test_clipped,
        filename="rf_actual_vs_predicted.png",
        save=args.save,
    )
    plot_actual_vs_predicted_garden_size(
        predicted=bdt_predictions,
        actual=y_test_clipped,
        filename="bdt_actual_vs_predicted.png",
        save=args.save,
    )

    plot_buckets_confusion_matrix(
        predicted=rf_predictions,
        actual=y_test_clipped,
        filename="rf_confusion_matrix.png",
        save=args.save,
    )
    plot_buckets_confusion_matrix(
        predicted=bdt_predictions,
        actual=y_test_clipped,
        filename="bdt_confusion_matrix.png",
        save=args.save,
    )

    if args.save:
        save_rf = config["output"]["model"]["garden_size_model"]["random_forest"]
        save_utils.save_model_to_pkl_s3(rf_model, save_rf)

        save_bdt = config["output"]["model"]["garden_size_model"][
            "boosted_decision_tree"
        ]
        save_utils.save_model_to_pkl_s3(bdt_model, save_bdt)
