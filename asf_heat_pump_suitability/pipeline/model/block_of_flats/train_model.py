import numpy as np
import pandas as pd
from typing import Iterable
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    HalvingRandomSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import f1_score

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


def train_block_of_flats_classifier(
    df: pl.DataFrame, id_col: str, features: Iterable[str], target: str
) -> RandomForestClassifier:
    """
    Args:
        df (pl.DataFrame): engineered features
        id_col (str): name of ID column
        features (Iterable[str]): features used to train the model
        target (str): name of target variable
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
    rfc = RandomForestClassifier(random_state=RNG)

    # Conduct halving random search for hyperparameters and cross-validation
    search = HalvingRandomSearchCV(
        estimator=rfc,
        param_distributions=PARAM_DISTRIBUTIONS,
        factor=2,  # Reduce halving aggressiveness
        random_state=RANDOM_STATE,
        cv=cv,  # Defaults to use StratifiedKFold splitter with 5 folds because model is binary classifier
        scoring="f1",  # Use F1 scoring metric for optimisation
        n_jobs=-1,  # Use all available cores
    ).fit(X_train, y_train)

    print(f"Best F1 score: {search.best_score_}")
    print(f"Best params: {search.best_params_}")

    # Train final classifier model on full training set with the selected hyperparameters
    final_model = RandomForestClassifier(**search.best_params_, random_state=RNG)
    final_model.fit(X_train, y_train)

    # Evaluate final model on hold out validation set
    y_pred = final_model.predict(X_test)
    val_f1 = f1_score(y_test, y_pred)
    print(f"F1 score on hold out validation set: {val_f1}")

    return final_model
