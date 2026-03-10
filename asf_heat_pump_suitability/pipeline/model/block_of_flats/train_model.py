"""
Functions to train and apply a random forest binary classifier model.

Contains script to train a random forest classifier to identify buildings as blocks of flats or not, given features derived
from building footprint and UPRN geospatial information.

To run the script:
asf_heat_pump_suitability/pipeline/model/block_of_flats/train_model.py

Set the required parameters:
- `uprns` takes a path to a parquet file containing domestic UPRNs with their X and Y coordinates.
- `labelled_data` takes a path to labelled data to train binary classification model on in parquet file format. Requires a boolean 'block_of_flats' column and a building
ID column with one row per building.

Note: `uprns` file must contain all domestic UPRNs within the area(s) that `labelled_data` samples from.

Pass the optional `save` parameter if saving to S3 is desired.
"""

import argparse
from typing import Iterable, Type

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    HalvingRandomSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.model_selection._search import BaseSearchCV

# Set random state int and RandomState instance
# See more on controlling randomness in sklearn docs: https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness
RANDOM_STATE = 8  # used in cross-validation for consistency of repeated calls to split dataset
RNG = np.random.RandomState(RANDOM_STATE)  # used in training random forest classifier to increase robustness

# Number of splits for StratifiedKFold cross-val strategy in parameter search
N_SPLITS = 5

# Size (proportion) of test set
TEST_SIZE = 0.2

# Create param distributions for hyperparameter search
PARAM_DISTRIBUTIONS = {
    "n_estimators": list(range(100, 1001, 10)),
    "max_depth": list(range(10, 101, 5)),
    "min_samples_split": list(range(2, 21, 1)),
    "min_samples_leaf": list(range(1, 11, 1)),
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini"],
}

# Model features
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


def train_eval_rfc_block_of_flats_classifier(
    df: pl.DataFrame,
    id_col: str,
    features: Iterable[str],
    target: str,
    param_search: Type[BaseSearchCV] = "default",
    scoring: str = "f1",
    **kwargs,
) -> RandomForestClassifier:
    """
    Train and evaluate RandomForestClassifier in binary classification to predict whether a building is a block of flats or not. Uses a search cross-validator to identify best
    hyperparameters and conduct cross validation. Model training and cross-validation is performed on 80% of the data in
    `df` with stratification. A final round of training is performed on the full 80% of training data using the best
    hyperparameters identified, and the model is tested on the 20% hold-out test set.

    Args:
        df (pl.DataFrame): engineered features with labelled target variable
        id_col (str): name of building ID column
        features (Iterable[str]): features used to train the model
        target (str): name of target variable
        param_search (Type[BaseSearchCV]): a class (not an instance) of `BaseSearchCV`, e.g. `HalvingRandomSearchCV` or `HalvingGridSearchCV` etc.
        Defaults to using `HalvingRandomSearchCV` which will create an instance of this class with selected custom arguments for `param_distributions`, `factor`, `cv`,
        and `n_jobs`, using F1 `scoring` metric. If using something other than the default option, kwargs for the selected `BaseSearchCV` class must be given, including param_distributions.
        However, note that `estimator` and `random_state` args will always default to RandomForestClassifier and global RANDOM_STATE, respectively, for consistency.
        scoring (str): `param_search` scoring metric used to evaluate predictions on the test set. Default "f1".
        **kwargs for selected `BaseSearchCV` if `param_search` not set to `default`. Note that any kwargs here will be ignored if `param_search` set to `default`.

    Returns:
        RandomForestClassifier: trained binary classifier model
    """
    # Sort model dataframe so that results are replicable
    pd_df = df.to_pandas().set_index(id_col).sort_values(id_col)
    X = pd_df[features]
    y = pd_df[target]

    # Keep a final hold out test set aside
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Create cross-validation splitter and classifier
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    estimator = RandomForestClassifier(random_state=RNG)
    random_state = RANDOM_STATE
    print(f"Training Random Forest binary classifier model with random state: {RANDOM_STATE}...")

    if param_search == "default":
        # Conduct halving random search for hyperparameters and cross-validation
        search = HalvingRandomSearchCV(
            estimator=estimator,
            param_distributions=PARAM_DISTRIBUTIONS,
            factor=2,  # Reduce halving aggressiveness
            random_state=RANDOM_STATE,
            cv=cv,  # Defaults to use StratifiedKFold splitter with 5 folds because model is binary classifier
            scoring=scoring,  # Use F1 scoring metric for optimisation
            n_jobs=-1,  # Use all available cores
        ).fit(X_train, y_train)

    else:
        search = param_search(
            estimator=estimator,
            random_state=random_state,
            **kwargs,
        ).fit(X_train, y_train)

    print(f"Best {scoring} score during cross-validation and hyperparameter search: {search.best_score_}")
    print(f"Best params: {search.best_params_}")

    # Train final classifier model on full training set with the selected hyperparameters
    final_model = RandomForestClassifier(**search.best_params_, random_state=RNG)
    final_model.fit(X_train, y_train)

    # Evaluate final model on hold out test set
    y_pred = final_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(f"{scoring} score on hold out test set: {score}")

    return final_model


def predict_class_block_of_flats(
    model: RandomForestClassifier,
    features_df: pl.DataFrame,
    labelled_df: pl.DataFrame,
    id_col: str,
) -> pl.DataFrame:
    """
    Predict binary class (block of flats / not) on buildings using trained Random Forest Classifier model. Features required:
        - n_UPRNs
        - n_flats
        - building_area_m2
        - building_perimeter_m
        - proportion_flats
        - UPRNs_per_building_m2
        - concave_hull_area_m2
        - uprns_per_hull_area_m2
        - flats_per_hull_area_m2
        - avg_n_stacked_uprns
        - std_n_stacked_uprns

    Args:
        model (RandomForestClassifier): trained model for binary classification of buildings into blocks of flats or not
        features_df (pl.DataFrame): buildings to predict classes on with engineered features for model
        labelled_df (pl.DataFrame): labelled training data with features and target variable
        id_col (str): name of building ID column in both `features_df` and `labelled_df` (must be the same)

    Returns:
        pl.DataFrame: one row per building with predicted class and probability of predicted class
    """
    print("Predicting classes (block of flats / not) and class probability of buildings...")
    concat_dfs = []
    labelled_ids = labelled_df[id_col].unique()
    X_df = (
        features_df.filter(pl.col("n_flats") > 0, ~pl.col(id_col).is_in(labelled_ids))
        .to_pandas()
        .set_index(id_col)[FEATURES]
    )
    predictions_df = X_df.copy()
    predictions_df["block_of_flats"] = model.predict(X_df)

    # Add probability of predictions
    predictions_df[f"block_of_flats_proba_{model.classes_[0]}"] = model.predict_proba(X_df)[:, 0]
    predictions_df[f"block_of_flats_proba_{model.classes_[1]}"] = model.predict_proba(X_df)[:, 1]

    # Combine into one probability label - final probability label indicates probability of the class assigned
    concat_dfs.append(
        pl.from_pandas(predictions_df, include_index=True).with_columns(
            pl.when(pl.col("block_of_flats"))
            .then(pl.col("block_of_flats_proba_True"))
            .when(~pl.col("block_of_flats"))
            .then(pl.col("block_of_flats_proba_False"))
            .alias("block_of_flats_label_proba")
        )
    )

    # Set manually labelled block probability to 1
    concat_dfs.append(
        labelled_df.filter(pl.col(id_col).is_in(labelled_ids)).with_columns(
            pl.lit(1.0).alias("block_of_flats_label_proba")
        )
    )

    # Add required columns to buildings which do not contain flats for concatenation
    concat_dfs.append(
        features_df.filter(pl.col("n_flats") == 0).with_columns(
            pl.lit(False).alias("block_of_flats"),
            pl.lit(None).alias("block_of_flats_label_proba"),
        )
    )

    cols = [id_col, "block_of_flats", "block_of_flats_label_proba"]

    return pl.concat([df.select(cols) for df in concat_dfs])


def extend_df_in_block_of_flats_label(
    uprns_df: pl.DataFrame, mapping: dict, predictions_df: pl.DataFrame, id_col: str
) -> pl.DataFrame:
    """
    Join predicted building class (block of flats / not) to the corresponding UPRNs located within.

    Args:
        uprns_df (pl.DataFrame): dataset with UPRN column
        mapping (dict): mapping of UPRNs (keys) to building IDs (values), where the building ID represents the building the UPRN is located within
        predictions_df (pl.DataFrame): one row per building with predicted class and probability of predicted class
        id_col (str): name of building ID column in `predictions_df`

    Returns:
        pl.DataFrame: one row per UPRN with `in_block_of_flats` label indicating the class of building the UPRN is located within
    """
    print("Adding `in_block_of_flats` label to UPRNs...")
    return (
        uprns_df.with_columns(
            # Map building IDs to the UPRNs they contain
            pl.col("UPRN").replace(mapping, default=None).alias(id_col)
        )
        # Join the predicted block of flats label to the UPRNs via the building ID
        .join(predictions_df, how="left", on=id_col)
        # Rename class for UPRN-level data
        .rename({"block_of_flats": "in_block_of_flats"})
    )


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--uprns",
        help="Path to domestic UPRN dataset with X and Y coordinates in parquet.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--labelled_data",
        help="Path to labelled data to train binary classification model on in parquet file format. Building ID column required.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save",
        help="Save trained model to S3.",
        type=bool,
        required=False,
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.getters import load_tree_input
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.model.block_of_flats import (
        feature_engineering,
    )
    from asf_heat_pump_suitability.pipeline.transform import uprns
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()

    # ------------------------ #
    # LOAD DATA
    # Load UPRN data
    print(f"Loading domestic UPRNs from: {args.uprns}")
    uprns_df = pl.read_parquet(args.uprns, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"])
    # Get geopoints of UPRNs
    uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

    # Load building footprint data
    # TODO scale beyond sampling areas
    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building",
        grid_squares=config["constant"]["grid_squares"]["sampling_areas"],
    )

    # ------------------------ #
    # IMPUTE PROPERTY TYPE FLAT
    # Create boolean column called `property_type_flat` to identify flats
    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
    uprns_gdf["property_type_flat"] = uprns_gdf["UPRN"].isin(flat_uprns)

    # ------------------------ #
    # FEATURE ENGINEERING
    building_features_df = feature_engineering.generate_df_features(
        buildings_gdf=building_footprints_gdf,
        uprns_gdf=uprns_gdf,
        id_col="ID",
    )

    # ------------------------ #
    # TRAIN MODEL
    labelled_df = pl.read_parquet(args.labelled_data)
    model_df = labelled_df.join(building_features_df, how="left", on="ID")

    model = train_eval_rfc_block_of_flats_classifier(
        df=model_df,
        id_col="ID",
        features=FEATURES,
        target="block_of_flats",
        param_search="default",
    )

    if args.save:
        save_as = config["output"]["save_as"]["block_of_flats_model"]
        save_utils.save_model_to_pkl_s3(model, save_as)
