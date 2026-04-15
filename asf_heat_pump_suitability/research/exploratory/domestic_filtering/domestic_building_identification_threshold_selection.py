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

from sklearn.metrics import matthews_corrcoef
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
        gpd.GeoDataFrame: buildings in area of interest labelled with actual and predicted domestic UPRN counts and boolean
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

    # UPRN density measures
    labelled_gdf["predicted_UPRN_per_m2"] = (
        labelled_gdf["predicted_UPRN_count"] / labelled_gdf["footprint_area_m2"]
    )
    labelled_gdf["total_UPRN_per_m2"] = (
        labelled_gdf["total_UPRN_count"] / labelled_gdf["footprint_area_m2"]
    )

    # UPRN density measures
    labelled_gdf["m2_per_predicted_UPRN"] = (
        labelled_gdf["footprint_area_m2"] / labelled_gdf["predicted_UPRN_count"]
    )
    labelled_gdf["m2_per_total_UPRN"] = (
        labelled_gdf["footprint_area_m2"] / labelled_gdf["total_UPRN_count"]
    )

    return labelled_gdf


def generate_df_threshold_evaluation(
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

            if mcc > max_mcc:
                max_mcc = mcc
                best_threshold = threshold

        # Calculate the number of false positives that are removed using best threshold
        N_removed_false_positives = len(
            model_df[
                # Reached threshold for non-domestic: labelled non-domestic
                (model_df[feature] > best_threshold)
                # Original pipeline mislabelled as domestic
                & (~model_df["actual_domestic"])
                & (model_df["predicted_domestic"])
            ]
        )

        # Calculate the number of true positives that are removed using best threshold
        N_removed_true_domestic = len(
            model_df[
                # Reached threshold for non-domestic: labelled non-domestic
                (model_df[feature] > best_threshold)
                # Original pipeline correctly identified as domestic
                & (model_df["actual_domestic"])
                & (model_df["predicted_domestic"])
            ]
        )

        # Store the best result for this feature
        scores["feature"].append(feature)
        scores["best_threshold"].append(best_threshold)
        scores["max_mcc"].append(max_mcc)
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


def extract_tuple_best_feature_threshold(df: pd.DataFrame) -> tuple:
    """
    Exract the best feature and corresponding threshold for implementation by maximising for Matthew's Correlation Coefficient.

    Args:
        df (pd.DataFrame): dataframe of features, best thresholds, and maximum MCC values

    Returns:
        tuple: feature name and best threshold value
    """
    # Get best threshold overall
    threshold = df[df["max_mcc"] == df["max_mcc"].max()]["best_threshold"].values[0]

    # Get best feature
    feature = df[df["max_mcc"] == df["max_mcc"].max()]["feature"].values[0]

    return feature, threshold


def plot_folium_threshold_effect_on_labelling(
    boundary: shapely.Polygon | shapely.MultiPolygon,
    labelled_gdf: gpd.GeoDataFrame,
    feature: str,
    threshold: float,
    uprns_gdf: gpd.GeoDataFrame,
) -> folium.Map:
    """
    Plot a folium map to show the effects on building-level labelling of implementing the best feature and threshold combination in the domestic
    identification pipeline.

    Args:
        labelled_gdf (gpd.GeoDataFrame): buildings in area of interest labelled with actual and predicted domestic UPRN counts and boolean.
        feature (str): name of feature to implement threshold in
        threshold (float): threshold above which buildings are classed as non-domestic
        uprns_gdf (gpd.GeoDataFrame): all UPRNs in area of interest and their corresponding point geometries

    Returns:
        folium.Map
    """
    # Get domestic EPC UPRNs and join to the building footprints to identify buildings containing domestic EPC records
    epc_uprns = uprns.load_set_valid_epc_uprns(epc_type="domestic")
    epc_gdf = uprns_gdf[uprns_gdf["UPRN"].isin(epc_uprns)][["UPRN", "geometry"]]
    labelled_gdf = (
        labelled_gdf.sjoin(epc_gdf, how="left", predicate="contains")
        .drop_duplicates(subset="ID")
        .astype({"UPRN": "bool"})
    )

    # Remove buildings containing a domestic EPC as these will be retained regardless of threshold
    labelled_gdf = labelled_gdf[~labelled_gdf["UPRN"]]

    # Still erroneously labelled as domestic
    false_positives_gdf = labelled_gdf[
        # Below the non-domestic threshold - predicted domestic
        (labelled_gdf[feature] <= threshold)
        # No council tax record - true not domestic
        & (~labelled_gdf["actual_domestic"])
    ]

    # Newly correctly removed
    true_negatives_gdf = labelled_gdf[
        # Reached the non-domestic threshold - predicted non-domestic
        (labelled_gdf[feature] > threshold)
        # No council tax record - true not domestic
        & (~labelled_gdf["actual_domestic"])
    ]

    # Newly falsely removed
    false_negatives_gdf = labelled_gdf[
        # Reached the non-domestic threshold - predicted non-domestic
        (labelled_gdf[feature] > threshold)
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
        config["data"]["processed"]["plymouth_residential_uprns"]
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
        "predicted_UPRN_per_m2",
        "total_UPRN_per_m2",
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

    # Find best thresholds for each feature
    thresholds_df = generate_df_threshold_evaluation(
        model_df=features_df, features=features
    )

    # Get best feature and threshold
    feature, threshold = extract_tuple_best_feature_threshold(thresholds_df)

    # Plot effect of threshold on the final labelling in the pipeline
    plot_folium_threshold_effect_on_labelling(
        boundary=plymouth_boundary,
        labelled_gdf=labelled_gdf,
        feature=feature,
        threshold=threshold,
        uprns_gdf=uprns_gdf,
    )

    # Sense check results with basic decision tree classifier
    clf = train_model_decision_tree_classifier(
        model_df=features_df,
        features=features,
        save_as="plymouth_decision_tree_nobalance",
    )
