"""
Exploratory script to test different features and thresholds within that feature to improve the pipeline to identify
buildings containing domestic UPRNs.

The script uses council tax data from Plymouth as a ground truth of domestic UPRNs to compare against and uses the
existing domestic identification pipeline outputs (as of 22 Apr 2026) as a baseline to improve upon. The aim of the
exploration is to remove more true non-domestic buildings from our predicted domestic dataset.

The analysis explores using ROC AUC score and Youden's J statistic; and Matthew's correlation coefficient (MCC) to identify
a feature and threshold. MCC was selected for final feature and threshold selection due to being a more robust metric
for highly imbalanced classification problems (seen here). MCC results for a single feature are compared to a
decision tree classifier with multiple input features to validate results. Although performance of the decision tree
classifier is slightly better, the single feature selected by MCC analysis was selected for pipeline improvement in
alpha testing due to explainability and time constraints for more thorough evaluation of the classifier.
"""

from pathlib import Path
import os

from typing import List
from collections import defaultdict

import geopandas as gpd
import pandas as pd
import polars as pl
import numpy as np
import shapely

import matplotlib.pyplot as plt
import folium

from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier, plot_tree

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import (
    base_getters,
    load_tree_input,
    load_boundaries,
    load_geodata,
)
from asf_heat_pump_suitability.pipeline.transform import uprns
from asf_heat_pump_suitability.utils import mapping_utils, plotting_utils


def transform_gdf_council_tax(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert council tax UPRN data into geodataframe with point geometries per UPRN. Note, this drops rows without UPRN
    coordinates.

    Args:
        df (pd.DataFrame): raw council tax UPRN data

    Returns:
        gpd.GeoDataFrame: council tax UPRNs with point geometries
    """
    # Remove empty UPRN and coordinate rows
    df = df[(df["UPRN"] != "") & (df["EASTING"] != "") & (df["NORTHING"] != "")]

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["EASTING"], df["NORTHING"]),
        crs="EPSG:27700",
    ).drop_duplicates(subset="UPRN")


def label_gdf_buildings_domestic_bool(
    buildings_gdf: gpd.GeoDataFrame,
    uprn_gdf: gpd.GeoDataFrame,
    council_tax_gdf: gpd.GeoDataFrame,
    pipeline_gdf: gpd.GeoDataFrame,
    boundary: shapely.Polygon | shapely.MultiPolygon,
) -> gpd.GeoDataFrame:
    """
    Label all buildings in area of interest with actual (true) and predicted domestic labels, using council tax data and
    current pipeline outputs, respectively. Label with actual and predicted domestic UPRN counts, as well as total UPRN
    count (domestic and non-domestic).

    Args:
        buildings_gdf (gpd.GeoDataFrame): building footprints for area of evaluation.
        uprn_gdf (gpd.GeoDataFrame): all UPRNs and point geometries for area of interest.
        council_tax_gdf (gpd.GeoDataFrame): processed council tax data with empty UPRN rows removed and with geopoints per UPRN.
        pipeline_gdf (gpd.GeoDataFrame): domestic UPRNs identified by current pipeline.
        boundary (shapely.Polygon | shapely.MultiPolygon): boundary of area of interest.

    Returns:
        gpd.GeoDataFrame: buildings in area of interest labelled with total UPRN, actual and predicted domestic UPRN
        counts and boolean flags for actual and predicted building containing domestic
    """
    # Select buildings in area of interest
    bounded_buildings_gdf = buildings_gdf[
        buildings_gdf["geometry"].intersects(boundary)
    ]

    # Label actual domestic buildings (buildings containing at least one council tax domestic UPRN)
    actual_domestic_df = (
        bounded_buildings_gdf.sjoin(
            council_tax_gdf[["UPRN", "geometry"]], how="inner", predicate="contains"
        )
        .drop(columns="index_right")
        .groupby("ID")
        .agg(actual_UPRN_count=("UPRN", "count"))
        .reset_index()
    )

    # Label current pipeline predictions for domestic buildings
    predicted_domestic_df = (
        bounded_buildings_gdf.sjoin(
            pipeline_gdf[["UPRN", "geometry"]], how="inner", predicate="contains"
        )
        .drop(columns="index_right")
        .groupby("ID")
        .agg(predicted_UPRN_count=("UPRN", "count"))
        .reset_index()
    )

    # Join all UPRNs to buildings - retain only buildings with UPRNs
    buildings_of_interest_df = (
        (
            bounded_buildings_gdf.sjoin(
                uprn_gdf[["UPRN", "geometry"]], how="inner", predicate="contains"
            )
        )
        .drop(columns="index_right")
        .groupby("ID")
        .agg(total_UPRN_count=("UPRN", "count"))
        .reset_index()
    )

    # Get data per building
    labelled_df = (
        buildings_of_interest_df.merge(actual_domestic_df, how="left", on="ID")
        .merge(predicted_domestic_df, how="left", on="ID")
        .fillna({"actual_UPRN_count": 0, "predicted_UPRN_count": 0})
    )

    # Create boolean labels at building level for actual and predicted buildings containing domestic UPRNs
    labelled_df["actual_domestic"] = np.where(
        labelled_df["actual_UPRN_count"] > 0, True, False
    )
    labelled_df["predicted_domestic"] = np.where(
        labelled_df["predicted_UPRN_count"] > 0, True, False
    )

    # Join building geometries back on
    return gpd.GeoDataFrame(
        labelled_df.merge(
            bounded_buildings_gdf[["ID", "geometry"]], how="inner", on="ID"
        ),
        geometry="geometry",
        crs="EPSG:27700",
    )


def generate_gdf_domestic_modelling_features(
    labelled_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Generate a set of basic building-level features to use for binary classification into 'contains domestic' or not.

    Args:
        labelled_gdf (gpd.GeoDataFrame): buildings in area of interest labelled with actual and predicted domestic UPRN counts and boolean.

    Returns:
        gpd.GeoDataFrame: basic building-level features for binary classification
    """
    # Building features
    labelled_gdf["footprint_area_m2"] = labelled_gdf.area
    labelled_gdf["building_perimeter_m"] = labelled_gdf.length

    # Area per UPRN measures
    labelled_gdf["m2_per_predicted_UPRN"] = (
        labelled_gdf["footprint_area_m2"] / labelled_gdf["predicted_UPRN_count"]
    )
    labelled_gdf["m2_per_total_UPRN"] = (
        labelled_gdf["footprint_area_m2"] / labelled_gdf["total_UPRN_count"]
    )

    return labelled_gdf


def generate_df_threshold_evaluation_mcc(
    model_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """
    For each feature in `features`, test a set of candidate thresholds to identify the best threshold for that feature
    for binary classification of buildings into 'contains domestic' or not. Candidate thresholds for each feature are
    the values at each percentile in that feature. The best threshold is selected by maximising for Matthew's Correlation
    Coefficient (MCC).

    Args:
        model_df (pd.DataFrame): dataframe with target variable and all features of interest
        features (List[str]): features of interest

    Returns:
        pd.DataFrame: best threshold and corresponding MCC for each feature with some summary statistics
    """
    # Create empty dict to collect scores and other summary information for each feature
    scores = defaultdict(list)
    y_true = model_df["actual_domestic"]

    # Calculate current number of false positives (true: non-domestic, pred: domestic)
    N_fp_before = len(
        model_df[(~model_df["actual_domestic"]) & (model_df["predicted_domestic"])]
    )

    # Calculate current number of true positives (true: domestic, pred: domestic)
    N_tp_before = len(
        model_df[(model_df["actual_domestic"]) & (model_df["predicted_domestic"])]
    )

    for feature in features:
        max_mcc = -1  # MCC ranges from -1 to 1
        best_threshold = None

        # Get unique values to test as candidate thresholds from percentile range
        candidate_thresholds = np.percentile(model_df[feature].dropna(), range(1, 100))

        # Calculate MCC for each threshold
        for threshold in candidate_thresholds:
            # Anything below the threshold is labelled domestic
            y_pred = np.where(model_df[feature] <= threshold, 1, 0)
            mcc = matthews_corrcoef(y_true, y_pred)

            # Check absolute MCC to account for thresholds that predict the reverse
            if abs(mcc) > max_mcc:
                max_mcc = abs(mcc)
                best_threshold = threshold
                # Positive MCC means anything below threshold labelled domestic
                if mcc > 0:
                    direction = "domestic_below_threshold"
                # Positive MCC means anything above threshold labelled domestic
                else:
                    direction = "domestic_above_threshold"

        _model_df = model_df.copy()
        if direction == "domestic_below_threshold":
            _model_df["new_predicted_domestic"] = _model_df[feature] <= best_threshold
        else:
            _model_df["new_predicted_domestic"] = _model_df[feature] > best_threshold

        # Calculate the number of false positives that are removed using best threshold
        N_removed_false_positives = len(
            _model_df[
                # Labelled non-domestic
                (~_model_df["new_predicted_domestic"])
                # Original pipeline mislabelled as domestic
                & (~_model_df["actual_domestic"])
                & (_model_df["predicted_domestic"])
            ]
        )

        # Calculate the number of true positives that are removed using best threshold
        N_removed_true_domestic = len(
            _model_df[
                # Labelled non-domestic
                (~_model_df["new_predicted_domestic"])
                # Original pipeline correctly identified as domestic
                & (_model_df["actual_domestic"])
                & (_model_df["predicted_domestic"])
            ]
        )

        # Store the best result for this feature
        scores["feature"].append(feature)
        scores["best_threshold"].append(best_threshold)
        scores["max_mcc"].append(max_mcc)
        scores["direction"].append(direction)
        scores["N_removed_false_positives"].append(N_removed_false_positives)
        scores["pc_removed_false_positives"].append(
            N_removed_false_positives / N_fp_before * 100
        )
        scores["N_removed_true_positives"].append(N_removed_true_domestic)
        scores["pc_removed_true_positives"].append(
            N_removed_true_domestic / N_tp_before * 100
        )

    scores_df = pd.DataFrame(scores).sort_values(by="max_mcc", ascending=False)
    print(scores_df)

    return scores_df


def generate_df_threshold_evaluation_roc_auc(
    model_df: pd.DataFrame, features: List[str]
) -> pl.DataFrame:
    """
    For each feature in `features`, calculate the ROC AUC score for that feature for binary classification of buildings
    into 'contains domestic' or not. Select the feature with the highest score, then calculate Youden's J statistic for
    candidate thresholds for that feature. The best threshold is selected by maximising for Youden's J statistic.

    Args:
        model_df (pd.DataFrame): dataframe with target variable and all features of interest
        features (List[str]): features of interest

    Returns:
        pl.DataFrame: best feature with Youden's J statistic for each threshold tested with some summary statistics
    """
    # Find best feature for thresholding from ROC AUC score
    scores = {
        "feature": features,
        "score": [
            roc_auc_score(model_df["actual_domestic"], model_df[feature])
            for feature in features
        ],
    }
    scores_df = pl.DataFrame(scores).with_columns(
        pl.when(pl.col("score") < 0.5)
        .then(1 - pl.col("score"))
        .otherwise(pl.col("score"))
        .alias("score"),
        pl.when(pl.col("score") < 0.5)
        .then(pl.lit("less_than"))
        .otherwise(pl.lit("greater_than"))
        .alias("direction_of_threshold"),
    )

    print("ROC AUC scores for each feature:")
    print(scores_df)

    best_score = scores_df.filter(pl.col("score") == pl.max("score"))
    best_feature = best_score["feature"][0]
    direction = best_score["direction_of_threshold"][0]
    print(
        f"\nBest ROC AUC score: {best_score['score'][0]}\nBest feature: {best_feature}"
    )

    # Calculate ROC curve and Youden's J statistic for the selected feature
    # sensitivity + specificity - 1 == TPR - FPR
    if direction == "greater_than":
        fpr, tpr, thresholds = roc_curve(
            y_true=model_df["actual_domestic"], y_score=model_df[best_feature]
        )

    else:
        fpr, tpr, thresholds = roc_curve(
            y_true=~model_df["actual_domestic"], y_score=model_df[best_feature]
        )

    youdens_df = pl.DataFrame({"youdens": tpr - fpr, "threshold": thresholds}).sort(
        by="youdens", descending=True
    )

    # Number of true non-domestic buildings removed which were mislabelled by pipeline as domestic
    n_removed_mislabeled_buildings = {
        threshold: len(
            model_df[
                # Buildings below the selected threshold are classed as non-domestic
                (model_df[best_feature] > threshold)
                & (~model_df["actual_domestic"])
                & (model_df["predicted_domestic"])
            ]
        )
        for threshold in youdens_df["threshold"]
    }
    n_removed_mislabeled_buildings_df = pl.DataFrame(
        {
            "threshold": n_removed_mislabeled_buildings.keys(),
            "N_removed_false_positives": n_removed_mislabeled_buildings.values(),
        }
    )

    # Number of true domestic buildings removed by threshold
    n_removed_true_domestic_buildings = {
        threshold: len(
            model_df[
                # Buildings below the selected threshold are classed as non-domestic
                (model_df[best_feature] < threshold)
                & (model_df["actual_domestic"])
            ]
        )
        for threshold in youdens_df["threshold"]
    }

    n_removed_true_domestic_buildings_df = pl.DataFrame(
        {
            "threshold": n_removed_true_domestic_buildings.keys(),
            "N_removed_true_positives": n_removed_true_domestic_buildings.values(),
        }
    )

    # Return results with summary statistics about buildings removed
    youdens_df = youdens_df.join(
        n_removed_mislabeled_buildings_df, how="left", on="threshold"
    ).join(n_removed_true_domestic_buildings_df, how="left", on="threshold")

    print(youdens_df)
    return youdens_df


def extract_tuple_best_feature_threshold(df: pd.DataFrame) -> dict:
    """
    Exract the best feature and corresponding threshold for implementation by maximising for Matthew's Correlation Coefficient.

    Args:
        df (pd.DataFrame): dataframe of features, best thresholds, and maximum MCC values

    Returns:
        tuple: feature name and best threshold value
    """
    best_row = df[df["max_mcc"] == df["max_mcc"].max()]

    # Get best threshold overall
    threshold = best_row["best_threshold"].values[0]

    # Get best feature
    feature = best_row["feature"].values[0]

    # Get best direction
    direction = best_row["direction"].values[0]

    return {"feature": feature, "threshold": threshold, "direction": direction}


def plot_folium_threshold_effect_on_labelling(
    boundary: shapely.Polygon | shapely.MultiPolygon,
    labelled_gdf: gpd.GeoDataFrame,
    feature: str,
    threshold: float,
    uprns_gdf: gpd.GeoDataFrame,
    domestic_below_threshold: bool = True,
) -> folium.Map:
    """
    Plot a folium map to show the effects on building-level labelling of implementing the best feature and threshold combination in the domestic
    identification pipeline.

    Args:
        labelled_gdf (gpd.GeoDataFrame): buildings in area of interest labelled with actual and predicted domestic UPRN counts and boolean.
        feature (str): name of feature to implement threshold in
        threshold (float): threshold for binary classification of buildings as domestic or non-domestic
        uprns_gdf (gpd.GeoDataFrame): all UPRNs in area of interest and their corresponding point geometries
        domestic_below_threshold (str): set to False if buildings greater than the specified threshold should be labelled domestic.

    Returns:
        folium.Map
    """
    # Get domestic EPC UPRNs and join to the building footprints to identify buildings containing domestic EPC records
    epc_uprns = uprns.load_set_valid_epc_uprns(epc_type="domestic")
    epc_gdf = uprns_gdf[uprns_gdf["UPRN"].isin(epc_uprns)][["UPRN", "geometry"]]
    labelled_gdf = (
        labelled_gdf.sjoin(epc_gdf, how="left", predicate="contains")
        .drop_duplicates(subset="ID")
        .fillna(False)
        .astype({"UPRN": "bool"})
        .rename(columns={"UPRN": "contains_domestic_EPC"})
    )

    if domestic_below_threshold:
        labelled_gdf["new_predicted_domestic"] = labelled_gdf[feature] <= threshold
    else:
        labelled_gdf["new_predicted_domestic"] = labelled_gdf[feature] > threshold

    # Remove buildings containing a domestic EPC from gdf as these will be retained regardless of threshold
    labelled_gdf = labelled_gdf[~labelled_gdf["contains_domestic_EPC"]]

    # Still erroneously labelled as domestic
    false_positives_gdf = labelled_gdf[
        # Predicted domestic
        (labelled_gdf["new_predicted_domestic"])
        # No council tax record - true not domestic
        & (~labelled_gdf["actual_domestic"])
    ]

    # Newly correctly removed
    true_negatives_gdf = labelled_gdf[
        # Predicted non-domestic
        (~labelled_gdf["new_predicted_domestic"])
        # No council tax record - true not domestic
        & (~labelled_gdf["actual_domestic"])
    ]

    # Newly falsely removed
    false_negatives_gdf = labelled_gdf[
        # Predicted non-domestic
        (~labelled_gdf["new_predicted_domestic"])
        # Council tax record - true domestic
        & (labelled_gdf["actual_domestic"])
    ]

    gdfs = {
        "False positives": false_positives_gdf,
        "True negatives": true_negatives_gdf,
        "False negatives": false_negatives_gdf,
    }

    colours = {
        "False positives": "red",
        "True negatives": "yellow",
        "False negatives": "blue",
    }

    return mapping_utils.plot_folium_polygon_map(
        boundary=boundary,
        gdf_dict=gdfs,
        colour_mapping=colours,
        popup_col=feature,
        save_as="plymouth_threshold_effect_on_labelling",
    )


def train_model_decision_tree_classifier(
    model_df: pd.DataFrame, features: List[str], save_as: str = None
) -> DecisionTreeClassifier:
    """
    Train a decision tree classifier model to label buildings as 'contains domestic' or not.

    Args:
        model_df (pd.DataFrame): dataframe with target variable and all features of interest
        features (List[str]): features of interest
        save_as (str): Path to save a plot of the decision tree to. Optional.
    """
    clf = DecisionTreeClassifier(max_leaf_nodes=5, class_weight=None)
    clf.fit(model_df[features], model_df["actual_domestic"])

    print(
        f"Matthew's Correlation Coefficient of Decision Tree Classifier: {matthews_corrcoef(model_df['actual_domestic'], clf.predict(model_df[features]))}"
    )

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_tree(clf, feature_names=features, class_names=True, ax=ax)

    if save_as:
        PROJECT_DIR = Path(__file__).resolve().parents[4]
        file_path = os.path.join(PROJECT_DIR, "outputs", "figures", f"{save_as}.png")
        plt.savefig(file_path)
    plt.close(fig)

    return clf


if __name__ == "__main__":
    # ------------------------------------------------------ #
    # LOAD DATASETS FOR ANALYSIS
    # ------------------------------------------------------ #
    # Council tax - ground truth
    raw_council_tax_uprns_df = pd.read_csv(
        config["data"]["geodata"]["council_tax_data"]["plymouth"]
    )
    council_tax_uprns_gdf = transform_gdf_council_tax(raw_council_tax_uprns_df)

    # Pipeline outputs
    pipeline_domestic_uprns_df = base_getters.load_df_from_s3(
        "s3://asf-heat-pump-suitability/local_heat_planning/static/domestic_identification/plymouth_residential_uprns.parquet"
    )
    pipeline_domestic_uprns_gdf = uprns.generate_gdf_uprn_coords(
        pipeline_domestic_uprns_df
    )

    # All building footprints
    buildings_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares="SX"
    )

    # Plymouth boundary
    plymouth_boundary = load_boundaries.load_gdf_local_authority_boundaries(
        select_las="Plymouth"
    )["geometry"].values[0]

    # All UPRNs with geometries
    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

    # ------------------------------------------------------ #
    # FIND FEATURE AND THRESHOLD TO IMPROVE FILTERING
    # ------------------------------------------------------ #
    # Label buildings with domestic / non-domestic flag
    labelled_gdf = label_gdf_buildings_domestic_bool(
        buildings_gdf=buildings_gdf,
        uprn_gdf=uprns_gdf,
        council_tax_gdf=council_tax_uprns_gdf,
        pipeline_gdf=pipeline_domestic_uprns_gdf,
        boundary=plymouth_boundary,
    )

    # We're only interested in buildings we've predicted to be containing domestic
    # This is because threshold will be applied AFTER the current pipeline
    labelled_gdf = labelled_gdf[labelled_gdf["predicted_domestic"]].copy()

    # Generate a set of basic features
    features_gdf = generate_gdf_domestic_modelling_features(labelled_gdf)
    features_df = features_gdf.drop(columns="geometry")
    features = [
        "total_UPRN_count",
        "predicted_UPRN_count",
        "footprint_area_m2",
        "building_perimeter_m",
        "m2_per_predicted_UPRN",
        "m2_per_total_UPRN",
    ]

    # Plot distribution of the features for each class
    plotting_utils.plot_feature_distribution_binary_classes(
        df=pl.from_pandas(features_df),
        features=features,
        target="actual_domestic",
        save_as="plymouth_feature_distribution_for_domestic_vs_non_domestic_buildings",
        density=True,
    )

    # Find best thresholds for each feature using MCC
    mcc_thresholds_df = generate_df_threshold_evaluation_mcc(
        model_df=features_df, features=features
    )

    # Find best thresholds for each feature using ROC AUC score and Youden's J statistic
    youdens_thresholds_df = generate_df_threshold_evaluation_roc_auc(
        model_df=features_df, features=features
    )

    # Get best feature, threshold, and direction
    best_combo = extract_tuple_best_feature_threshold(mcc_thresholds_df)

    # Plot effect of threshold on the final labelling in the pipeline
    plot_folium_threshold_effect_on_labelling(
        boundary=plymouth_boundary,
        labelled_gdf=labelled_gdf,
        feature=best_combo["feature"],
        threshold=best_combo["threshold"],
        uprns_gdf=uprns_gdf,
    )

    # Sense check results with basic decision tree classifier
    clf = train_model_decision_tree_classifier(
        model_df=features_df,
        features=features,
        save_as="plymouth_decision_tree_nobalance",
    )
