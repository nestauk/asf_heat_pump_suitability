import numpy as np
import polars as pl
import pandas as pd
from typing import Iterable, Type
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    train_test_split,
    HalvingRandomSearchCV,
    StratifiedKFold,
)
from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics import f1_score
import argparse

# Set random state int and RandomState instance
RANDOM_STATE = 8
RNG = np.random.RandomState(RANDOM_STATE)

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


def train_block_of_flats_classifier(
    df: pl.DataFrame,
    id_col: str,
    features: Iterable[str],
    target: str,
    param_search: Type[BaseSearchCV] = "default",
    scoring: str = "f1",
    **kwargs,
) -> RandomForestClassifier:
    """
    Train and evaluate RandomForestClassifier in binary classification. Uses a search cross-validator to identify best
    hyperparameters and conduct cross validation. Model training and cross-validation is performed on 80% of the data in
    `df` with stratification. A final round of training is performed on the full 80% of training data using the best
    hyperparameters identified, and the model is tested on the 20% hold-out test set.

    Args:
        df (pl.DataFrame): engineered features with labelled target variable
        id_col (str): name of ID column
        features (Iterable[str]): features used to train the model
        target (str): name of target variable,
        param_search (Type[BaseSearchCV]): a class (not an instance) of `BaseSearchCV`, e.g. `HalvingRandomSearchCV` or `HalvingGridSearchCV` etc.
        Defaults to using `HalvingRandomSearchCV` which will create an instance of this class with selected custom arguments for `param_distributions`, `factor`, `cv`,
        and `n_jobs`, using F1 `scoring` metric. If using something other than the default option, kwargs for the selected `BaseSearchCV` class must be given, including param_distributions.
        However, note that `estimator` and `random_state` args will always default to RandomForestClassifier and global RANDOM_STATE, respectively, for consistency.
        scoring (str): kwarg for `param_search` scoring metric used to evaluate predictions on the test set. Default "f1".
        **kwargs for selected `BaseSearchCV` if `param_search` not set to `default`. Note that any kwargs here will be ignored if `param_search` set to `default`.

    Returns:
        RandomForestClassifier: trained binary classifier model
    """
    # Sort model dataframe so that results are replicable
    pd_df = df.to_pandas().set_index(id_col).sort_values(id_col)
    X = pd_df[features]
    y = pd_df[target]

    # Keep a final hold out validation set aside
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Create cross-validation splitter and classifier
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if "estimator" in locals() or "random_state" in locals():
        print(
            f"Training Random Forest Classifier model with random state: {RANDOM_STATE}..."
        )
    estimator = RandomForestClassifier(random_state=RNG)
    random_state = RANDOM_STATE

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

    print(f"Best {scoring} score: {search.best_score_}")
    print(f"Best params: {search.best_params_}")

    # Train final classifier model on full training set with the selected hyperparameters
    final_model = RandomForestClassifier(**search.best_params_, random_state=RNG)
    final_model.fit(X_train, y_train)

    # Evaluate final model on hold out test set
    y_pred = final_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(f"{scoring} score on hold out validation set: {score}")

    return final_model


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
        help="Labelled data to train binary classification model on in parquet file format. Building ID required.",
        type=str,
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.getters import load_tree_input
    from asf_heat_pump_suitability.pipeline.transform import uprns
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.model.block_of_flats import (
        feature_engineering,
    )
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()

    # ------------------------ #
    # LOAD DATA
    # Load UPRN data
    print(f"Loading domestic UPRNs from: {args.uprns}")
    uprns_df = pl.read_parquet(
        args.uprns, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"]
    )
    # Get geopoints of UPRNs
    uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

    # Load building footprint data
    # TODO scale beyond Plymouth
    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares="SX"
    )

    # ------------------------ #
    # IMPUTE PROPERTY TYPE FLAT
    # Create boolean column called `property_type_flat` to identify flats
    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
    uprns_gdf["property_type_flat"] = uprns_gdf[uprns_gdf["UPRN"].isin(flat_uprns)]

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

    model = train_block_of_flats_classifier(
        df=model_df,
        id_col="ID",
        features=FEATURES,
        target="block_of_flats",
        param_search="default",
    )

    save_as = config["output"]["save_as"]["block_of_flats_model"]
    save_utils.save_model_to_pkl_s3(model, save_as)
