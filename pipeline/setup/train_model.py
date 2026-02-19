"""Train the block-of-flats random forest binary classifier.

Trains a random forest classifier to identify buildings as blocks of flats or
not, given features derived from building footprint and UPRN geospatial data.
The trained model is saved as a pickle file to S3.

Run:
    python pipeline/setup/train_model.py \\
        --uprns s3://asf-heat-pump-suitability/.../domestic_uprns.parquet \\
        --labelled_data s3://asf-heat-pump-suitability/.../labelled_buildings.parquet

Required columns in --labelled_data:
    ID (str)              Building ID matching OS OpenMap Local building ID
    block_of_flats (bool) True if the building is a block of flats
"""

import argparse
import logging

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import f1_score
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, train_test_split

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import load_tree_input
from asf_heat_pump_suitability.pipeline.impute import property_type
from asf_heat_pump_suitability.pipeline.transform import uprns
from asf_heat_pump_suitability.utils import save_utils
from pipeline.model.block_of_flats import feature_engineering

logger = logging.getLogger(__name__)

RANDOM_STATE = 8
RNG = np.random.RandomState(RANDOM_STATE)
N_SPLITS = 5
TEST_SIZE = 0.2

PARAM_DISTRIBUTIONS = {
    "n_estimators": list(range(100, 1001, 10)),
    "max_depth": list(range(10, 101, 5)),
    "min_samples_split": list(range(2, 21, 1)),
    "min_samples_leaf": list(range(1, 11, 1)),
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini"],
}

FEATURES = [
    "n_UPRNs",
    "n_flats",
    "building_area_m2",
    "building_perimeter_m",
    "proportion_flats",
    "UPRNs_per_building_m2",
    "concave_hull_area_m2",
    "uprns_per_hull_area_m2",
    "flats_per_hull_area_m2",
    "avg_n_stacked_uprns",
    "std_n_stacked_uprns",
]


def train_block_of_flats_classifier(df: pl.DataFrame, id_col: str = "ID") -> RandomForestClassifier:
    """Train and evaluate a RandomForestClassifier to predict blocks of flats.

    Uses halving random search CV for hyperparameter optimisation (80/20 split,
    stratified). Returns the model trained on the full 80% training set with
    best hyperparameters.

    Args:
        df: Feature-engineered DataFrame with a ``block_of_flats`` boolean target.
        id_col: Building ID column name. Default "ID".

    Returns:
        RandomForestClassifier: Trained binary classifier.
    """
    pd_df = df.to_pandas().set_index(id_col).sort_values(id_col)
    X = pd_df[FEATURES]  # noqa: N806 — ML convention for feature matrix
    y = pd_df["block_of_flats"]

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    estimator = RandomForestClassifier(random_state=RNG)

    logger.info(f"Training Random Forest classifier (random_state={RANDOM_STATE})...")
    search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        factor=2,
        random_state=RANDOM_STATE,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    ).fit(X_train, y_train)

    logger.info(f"Best CV F1: {search.best_score_:.4f}  |  params: {search.best_params_}")

    final_model = RandomForestClassifier(**search.best_params_, random_state=RNG)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    logger.info(f"Hold-out test F1: {f1_score(y_test, y_pred):.4f}")

    return final_model


def parse_arguments() -> argparse.Namespace:
    """Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated Namespace.
    """
    parser = argparse.ArgumentParser(description="Train block-of-flats classifier and save to S3.")
    parser.add_argument("--uprns", help="Path to domestic UPRN parquet file.", type=str, required=True)
    parser.add_argument(
        "--labelled_data",
        help="Path to labelled building parquet file with 'ID' and 'block_of_flats' columns.",
        type=str,
        required=True,
    )
    parser.add_argument("--no_save", help="Skip saving the model to S3.", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    # Load domestic UPRNs and build GeoDataFrame
    logger.info(f"Loading domestic UPRNs from: {args.uprns}")
    uprns_df = pl.read_parquet(args.uprns, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"])
    uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

    # Load building footprints for sampling areas
    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building",
        grid_squares=config["constant"]["grid_squares"]["sampling_areas"],
    )

    # Impute flat property type
    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
    uprns_gdf["property_type_flat"] = uprns_gdf["UPRN"].isin(flat_uprns)

    # Engineer features
    building_features_df = feature_engineering.generate_df_features(
        buildings_gdf=building_footprints_gdf,
        uprns_gdf=uprns_gdf,
        id_col="ID",
    )

    # Load labelled data and join features
    labelled_df = pl.read_parquet(args.labelled_data)
    model_df = labelled_df.join(building_features_df, how="left", on="ID")

    # Train model
    model = train_block_of_flats_classifier(df=model_df)

    if not args.no_save:
        save_path = config["output"]["save_as"]["block_of_flats_model"]
        save_utils.save_model_to_pkl_s3(model, save_path)
