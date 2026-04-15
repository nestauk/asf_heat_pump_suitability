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


def calculate_dict_uprn_diffs_per_dataset(
    raw_council_tax_df, council_tax_gdf, pipeline_gdf
):
    council_uprns = set(council_tax_gdf["UPRN"])
    pipeline_uprns = set(pipeline_gdf["UPRN"])

    n_council_uprns = council_tax_gdf["UPRN"].nunique()
    n_pipeline_uprns = pipeline_gdf["UPRN"].nunique()

    results = {
        # Number of unique records / UPRNs per dataset
        "N unique records in council tax data": raw_council_tax_df["PROPREF"].nunique(),
        "N unique UPRNs in council tax data": n_council_uprns,
        "N unique UPRNs in pipeline output data": n_pipeline_uprns,
        # Differences in council tax UPRNs versus pipeline
        "N diff UPRNs in pipeline minus council tax": n_pipeline_uprns
        - n_council_uprns,
        "Proportion diff UPRNs in pipeline minus council tax": round(
            (n_pipeline_uprns - n_council_uprns) / n_pipeline_uprns, 3
        ),
        "N domestic UPRNs missing from pipeline": len(
            council_uprns.difference(pipeline_uprns)
        ),
        "Proportion domestic UPRNs missing from pipeline": round(
            len(council_uprns.difference(pipeline_uprns)) / n_council_uprns, 3
        ),
        "N UPRNs in pipeline but not council tax": len(
            pipeline_uprns.difference(council_uprns)
        ),
        "Proportion UPRNs in pipeline not in council tax": round(
            len(pipeline_uprns.difference(council_uprns)) / n_pipeline_uprns, 3
        ),
    }

    return results


def calculate_dict_building_diffs_per_dataset(
    n_council_records: int,
    council_tax_gdf: gpd.GeoDataFrame,
    pipeline_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
):
    council_buildings_gdf = buildings_gdf.sjoin(
        council_tax_gdf, how="inner", predicate="contains"
    ).drop(columns="index_right")

    pipeline_buildings_gdf = buildings_gdf.sjoin(
        pipeline_gdf, how="inner", predicate="contains"
    ).drop(columns="index_right")

    council_buildings = set(council_buildings_gdf["ID"])
    pipeline_buildings = set(pipeline_buildings_gdf["ID"])

    n_council_buildings = council_buildings_gdf["ID"].nunique()
    n_pipeline_buildings = pipeline_buildings_gdf["ID"].nunique()

    results = {
        "Proportion of council records located in building footprints": round(
            council_buildings_gdf["UPRN"].nunique() / n_council_records, 3
        ),
        "Proportion of council UPRNs located in building footprints": round(
            council_buildings_gdf["UPRN"].nunique() / len(council_tax_gdf), 3
        ),
        "Proportion of pipeline UPRNs located in building footprints": round(
            pipeline_buildings_gdf["UPRN"].nunique() / len(pipeline_gdf), 3
        ),
        "Proportion of council buildings in pipeline buildings": round(
            len(council_buildings.intersection(pipeline_buildings))
            / n_council_buildings,
            3,
        ),
        "N pipeline buildings containing a council tax UPRN": len(
            pipeline_buildings.intersection(council_buildings)
        ),
        "Proportion of pipeline buildings containing a council tax UPRN": round(
            len(pipeline_buildings.intersection(council_buildings))
            / n_pipeline_buildings,
            3,
        ),
    }

    return results


def generate_gdf_erroneous_pipeline_buildings(
    council_tax_gdf: gpd.GeoDataFrame,
    pipeline_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
):
    council_buildings_gdf = buildings_gdf.sjoin(
        council_tax_gdf, how="inner", predicate="contains"
    ).drop(columns="index_right")
    pipeline_buildings_gdf = buildings_gdf.sjoin(
        pipeline_gdf, how="inner", predicate="contains"
    ).drop(columns="index_right")

    council_buildings = set(council_buildings_gdf["ID"])

    false_positives_gdf = pipeline_buildings_gdf[
        ~pipeline_buildings_gdf["ID"].isin(council_buildings)
    ]
    uprns_per_building_gdf = false_positives_gdf.groupby("ID").agg(
        UPRN_count=("UPRN", "nunique"), geometry=("geometry", "first")
    )

    return gpd.GeoDataFrame(
        uprns_per_building_gdf, geometry="geometry", crs="EPSG:27700"
    )


def label_gdf_buildings_domestic_bool(
    buildings_gdf: gpd.GeoDataFrame,
    uprn_gdf: gpd.GeoDataFrame,
    council_tax_gdf: gpd.GeoDataFrame,
    pipeline_gdf: gpd.GeoDataFrame,
    boundary: shapely.Polygon | shapely.MultiPolygon,
):
    bounded_buildings_gdf = buildings_gdf[
        buildings_gdf["geometry"].intersects(boundary)
    ]

    # Label actual domestic buildings (buildings containing at least one domestic UPRN)
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

    labelled_df["actual_domestic"] = np.where(
        labelled_df["actual_UPRN_count"] > 0, True, False
    )
    labelled_df["predicted_domestic"] = np.where(
        labelled_df["predicted_UPRN_count"] > 0, True, False
    )

    return gpd.GeoDataFrame(
        labelled_df.merge(
            bounded_buildings_gdf[["ID", "geometry"]], how="inner", on="ID"
        ),
        geometry="geometry",
        crs="EPSG:27700",
    )


def generate_gdf_features_for_filtering(
    labelled_gdf,
):
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


def generate_df_threshold_evaluation(model_df: pd.DataFrame, features: List[str]):
    scores = defaultdict(list)
    y_true = model_df["actual_domestic"]

    for feature in features:
        max_mcc = -1  # MCC ranges from -1 to 1
        best_threshold = None

        # Get unique values to test as candidate thresholds from percentile range
        candidate_thresholds = np.percentile(model_df[feature].dropna(), range(1, 100))

        for threshold in candidate_thresholds:
            # Anything below the threshold is labelled domestic
            y_pred = np.where(model_df[feature] <= threshold, 1, 0)
            mcc = matthews_corrcoef(y_true, y_pred)

            if mcc > max_mcc:
                max_mcc = mcc
                best_threshold = threshold

        N_fp_before = len(
            model_df[(~model_df["actual_domestic"]) & (model_df["predicted_domestic"])]
        )

        N_removed_false_positives = len(
            model_df[
                (model_df[feature] > best_threshold)
                & (~model_df["actual_domestic"])
                & (model_df["predicted_domestic"])
            ]
        )

        N_tp_before = len(
            model_df[(model_df["actual_domestic"]) & (model_df["predicted_domestic"])]
        )

        N_removed_true_domestic = len(
            model_df[
                (model_df[feature] > best_threshold)
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


def extract_tuple_best_feature_threshold(df):
    # Get best threshold overall
    threshold = df[df["max_mcc"] == df["max_mcc"].max()]["best_threshold"].values[0]

    # Get best feature
    feature = df[df["max_mcc"] == df["max_mcc"].max()]["feature"].values[0]

    return (feature, threshold)


def plot_folium_threshold_effect_on_labelling(
    boundary, labelled_gdf, feature, threshold, uprns_gdf
):
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

    mapping_utils.plot_folium_polygon_map(
        boundary=boundary,
        gdf_dict=gdfs,
        colour_mapping=colours,
        popup_col=feature,
        save_as="plymouth_threshold_effect_on_labelling",
    )


def train_model_decision_tree_classifier(
    model_df: pd.DataFrame, features: List[str], save_as: str
):
    clf = DecisionTreeClassifier(max_leaf_nodes=5, class_weight=None)
    clf.fit(model_df[features], model_df["actual_domestic"])
    print(
        f"Matthew's Correlation Coefficient of Decision Tree Classifier: {matthews_corrcoef(model_df['actual_domestic'], clf.predict(model_df[features]))}"
    )

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

    # Council tax - ground truth
    raw_council_tax_uprns_gdf = pd.read_csv(
        config["data"]["geodata"]["council_tax_data"]["plymouth"]
    )

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

    # ------------------------------------------------------------------- #
    # ANALYSIS - COMPARISON OF CURRENT PIPELINE WITH COUNCIL TAX DATA
    # ------------------------------------------------------------------- #
    n_council_records = len(raw_council_tax_uprns_gdf)
    council_tax_uprns_gdf = transform_gdf_council_tax(raw_council_tax_uprns_gdf)

    results = calculate_dict_uprn_diffs_per_dataset(
        raw_council_tax_uprns_gdf, council_tax_uprns_gdf, pipeline_domestic_uprns_gdf
    )
    buildings_results = calculate_dict_building_diffs_per_dataset(
        n_council_records=n_council_records,
        council_tax_gdf=council_tax_uprns_gdf,
        pipeline_gdf=pipeline_domestic_uprns_gdf,
        buildings_gdf=buildings_gdf,
    )

    false_positives_gdf = generate_gdf_erroneous_pipeline_buildings(
        council_tax_gdf=council_tax_uprns_gdf,
        pipeline_gdf=pipeline_domestic_uprns_gdf,
        buildings_gdf=buildings_gdf,
    )
    mapping_utils.plot_folium_polygon_map(
        boundary=plymouth_boundary,
        gdf_dict={"False positives": false_positives_gdf},
        colour_mapping={"False positives": "red"},
        popup_col="UPRN_count",
        save_as="plymouth_false_positive_domestic_buildings",
    )

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
    features_gdf = generate_gdf_features_for_filtering(labelled_gdf)
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
