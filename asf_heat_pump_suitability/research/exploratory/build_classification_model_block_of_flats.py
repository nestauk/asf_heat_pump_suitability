# %% [markdown]
# ## Build binary classification model for blocks of flats
#
# This notebook trains a model for the binary classification of buildings into blocks of flats or not. In it we:
# - Load building and UPRN data for Plymouth and sample areas
# - Create features for each building
# - Load and preprocess a manually labelled sample of data
# - Plot features for labelled data
# - Train and evaluate binary classification model

# %%
import geopandas as gpd
import pandas as pd

import simplekml

import polars as pl
import numpy as np
import math

import matplotlib.pyplot as plt

import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    HalvingRandomSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    PrecisionRecallDisplay,
    recall_score,
    f1_score,
    precision_score,
)

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.transform import uprns
from asf_heat_pump_suitability.getters import load_tree_input

plt.style.use("tableau-colorblind10")

# %% [markdown]
# ## Load required data

# %% [markdown]
# ### Load Plymouth UPRN and building data

# %%
# Load Plymouth buildings and residential UPRNs with property type label
apr_buildings_plymouth_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/inputs/v042025_OSOpenMapLocal_geometries_selected/SX/SX_Building.shp"
)
oct_buildings_plymouth_gdf = gpd.read_file(
    "s3://asf-heat-pump-suitability/local_heat_planning/inputs/geodata/v202510_OSOpenMapLocal_geometries_selected/SX/SX_Building.shp"
)
plymouth_uprns_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/plymouth_residential_uprns_with_flats.parquet"
)

# Match buildings from April and October versions of the OS OpenMap Local Buildings file
apr_buildings_plymouth_gdf["geometry"] = apr_buildings_plymouth_gdf.normalize()
oct_buildings_plymouth_gdf["geometry"] = oct_buildings_plymouth_gdf.normalize()

apr_oct_buildings_gdf = apr_buildings_plymouth_gdf.merge(
    oct_buildings_plymouth_gdf, how="left", on="geometry", suffixes=["_apr", "_oct"]
)

# Create mapping of April and October IDs
apr_id = apr_oct_buildings_gdf["ID_apr"].tolist()
oct_id = apr_oct_buildings_gdf["ID_oct"].tolist()
apr_to_oct_id = dict(zip(apr_id, oct_id))
apr_buildings_plymouth_gdf["oct25_building_id"] = apr_buildings_plymouth_gdf["ID"].map(
    apr_to_oct_id.get
)

apr_buildings_plymouth_gdf = apr_buildings_plymouth_gdf.rename(
    columns={"ID": "apr25_building_id"}
)

# Join buildings to UPRNs
plymouth_uprns_gdf = (
    uprns.generate_gdf_uprn_coords(df=plymouth_uprns_df)
    .sjoin(apr_buildings_plymouth_gdf, how="inner", predicate="within")
    .drop(columns=["index_right", "FEATCODE"])
)

# Join UPRNs to buildings
buildings_w_uprns_gdf = apr_buildings_plymouth_gdf.sjoin(
    uprns.generate_gdf_uprn_coords(df=plymouth_uprns_df),
    how="left",
    predicate="contains",
).drop(columns=["index_right"])

# %% [markdown]
# ### Load UPRN and building data for other sampling areas (Nottingham, Bradford, Glasgow, Manchester, Bath)

# %%
# Load OS OpenMap Local Buildings layer for sampling areas
sampling_areas_buildings_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
    layer="building", grid_squares=["NS", "SD", "SE", "SJ", "SK", "ST"]
)

# Load residential UPRNs with property type label
sampling_areas_uprns_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/local_heat_planning/outputs/sampling_areas_residential_uprns_with_flats.parquet"
)

# Join buildings to UPRNs
sampling_areas_uprns_gdf = (
    uprns.generate_gdf_uprn_coords(df=sampling_areas_uprns_df)
    .sjoin(sampling_areas_buildings_gdf, how="inner", predicate="within")
    .drop(columns=["index_right", "FEATCODE"])
)

# Join UPRNs to buildings
sampling_areas_buildings_w_uprns_gdf = sampling_areas_buildings_gdf.sjoin(
    uprns.generate_gdf_uprn_coords(df=sampling_areas_uprns_df),
    how="left",
    predicate="contains",
).drop(columns=["index_right"])

# %% [markdown]
# ## Preprocess building and UPRN data and create model features

# %%
# Prepare sampling area data for concatenation
sampling_areas_buildings_w_uprns_gdf["apr25_building_id"] = None
sampling_areas_buildings_w_uprns_gdf = sampling_areas_buildings_w_uprns_gdf.rename(
    columns={"ID": "oct25_building_id"}
)
sampling_areas_buildings_w_uprns_gdf = sampling_areas_buildings_w_uprns_gdf[
    buildings_w_uprns_gdf.columns
]

# Concatenate buildings with UPRNs data
buildings_w_uprns_gdf = pd.concat(
    [buildings_w_uprns_gdf, sampling_areas_buildings_w_uprns_gdf]
)

# Create features from building data
buildings_w_uprns_gdf = buildings_w_uprns_gdf[~buildings_w_uprns_gdf["UPRN"].isna()]
buildings_w_uprns_gdf["building_area_m2"] = buildings_w_uprns_gdf.area
buildings_w_uprns_gdf["building_perimeter_m"] = buildings_w_uprns_gdf.length
buildings_w_uprns_df = pl.from_pandas(buildings_w_uprns_gdf.drop(columns=["geometry"]))

agg_building_df = (
    buildings_w_uprns_df.group_by("oct25_building_id")
    .agg(
        pl.col("apr25_building_id").first().name.keep(),
        pl.col("UPRN").count().alias("n_UPRNs"),
        pl.col("property_type_flat").sum().alias("n_flats"),
        pl.col("building_area_m2").first().name.keep(),
        pl.col("building_perimeter_m").first().name.keep(),
    )
    .filter(pl.col("n_flats") > 0)
    .with_columns(
        (pl.col("n_flats") / pl.col("n_UPRNs")).alias("proportion_flats"),
        (pl.col("n_UPRNs") / pl.col("building_area_m2")).alias("UPRNs_per_building_m2"),
    )
)

# %%
# Concatenate UPRNs with buildings data
sampling_areas_uprns_gdf["apr25_building_id"] = None
sampling_areas_uprns_gdf = sampling_areas_uprns_gdf.rename(
    columns={"ID": "oct25_building_id"}
)

all_uprns_gdf = pd.concat(
    [plymouth_uprns_gdf, sampling_areas_uprns_gdf[plymouth_uprns_gdf.columns]]
)

# Create concave hull feature to represent spatial distribution of UPRNs in each building
hull_gdf = all_uprns_gdf.dissolve("oct25_building_id").concave_hull().reset_index()
hull_gdf = hull_gdf.rename(columns={0: "geometry"}).set_geometry("geometry")
hull_gdf["concave_hull_area_m2"] = hull_gdf.area

# %%
# Join building features with concave hull feature
agg_building_df = agg_building_df.join(
    pl.from_pandas(hull_gdf.drop(columns="geometry")),
    how="left",
    on="oct25_building_id",
)

# Calculate additional features from the concave hull area
agg_building_df = agg_building_df.with_columns(
    (pl.col("n_UPRNs") / pl.col("concave_hull_area_m2")).alias(
        "uprns_per_hull_area_m2"
    ),
    (pl.col("n_flats") / pl.col("concave_hull_area_m2")).alias(
        "flats_per_hull_area_m2"
    ),
).with_columns(
    # UPRNs or flats per hull area can be infinite if all UPRNs/flats share the same coordinates (i.e. area = 0m2)
    # We change this to -1 for the model
    pl.when(pl.col("uprns_per_hull_area_m2").is_infinite())
    .then(-1)
    .otherwise(pl.col("uprns_per_hull_area_m2"))
    .alias("uprns_per_hull_area_m2"),
    pl.when(pl.col("flats_per_hull_area_m2").is_infinite())
    .then(-1)
    .otherwise(pl.col("flats_per_hull_area_m2"))
    .alias("flats_per_hull_area_m2"),
)

# %%
# Get count of UPRNs at each X and Y coordinates to get the count of UPRNs which share an exact location
all_uprns_df = pl.from_pandas(all_uprns_gdf.drop(columns="geometry"))
all_uprns_df = all_uprns_df.with_columns(
    n_stacked_uprns=pl.col("UPRN").count().over(["X_COORDINATE", "Y_COORDINATE"])
)

# Group by building and get the average and STD of UPRNs sharing the same coordinates
agg_uprns_df = all_uprns_df.group_by("oct25_building_id").agg(
    pl.col("apr25_building_id").first().name.keep(),
    pl.col("n_stacked_uprns").mean().alias("avg_n_stacked_uprns"),
    pl.col("n_stacked_uprns").std().alias("std_n_stacked_uprns"),
)

# Join all the calculated features together
features_df = agg_building_df.join(
    agg_uprns_df.select(
        ["oct25_building_id", "avg_n_stacked_uprns", "std_n_stacked_uprns"]
    ),
    how="left",
    on="oct25_building_id",
)

# %% [markdown]
# ## Load and process labelled training data

# %%
# Group the manually labelled archetypes into block of flats or not
block_of_flats_labels = ["BF", "BP", "SF"]
non_block_labels = ["WB", "TE", "TF", "CO", "OF", "TT"]

# %%
labelled_datasets = dict()

labelled_datasets["apr25"] = {
    "reindeer_1": "s3://asf-heat-pump-suitability/local_heat_planning/labelling/labelled/CORRECTED_plymouth_buildings_containing_flats_sample_n200_seed10.kml",
    "reindeer_2": "s3://asf-heat-pump-suitability/local_heat_planning/labelling/labelled/CORRECTED_small_plymouth_buildings_containing_flats_sample_n100_seed10.kml",
}

labelled_datasets["oct25"] = {
    "reindeer_3": "s3://asf-heat-pump-suitability/local_heat_planning/labelling/labelled/CORRECTED_20251121_reindeer_LABELLED_buildings_containing_flats_sample_n147_seed10.kml",
    "raccoon": "s3://asf-heat-pump-suitability/local_heat_planning/labelling/labelled/CORRECTED_20251121_raccoon_LABELLED_buildings_containing_flats_sample_n147_seed10.kml",
    "anteater": "s3://asf-heat-pump-suitability/local_heat_planning/labelling/labelled/CORRECTED_20251121_anteater_LABELLED_buildings_containing_flats_sample_n147_seed10.kml",
    "leopard": "s3://asf-heat-pump-suitability/local_heat_planning/labelling/labelled/CORRECTED_20251121_leopard_LABELLED_buildings_containing_flats_sample_n147_seed10.kml",
    "springbok": "s3://asf-heat-pump-suitability/local_heat_planning/labelling/labelled/CORRECTED_20251121_springbok_LABELLED_buildings_containing_flats_sample_n147_seed10.kml",
    "yak": "s3://asf-heat-pump-suitability/local_heat_planning/labelling/labelled/CORRECTED_20251121_yak_LABELLED_buildings_containing_flats_sample_n147_seed10.kml",
}


# %%
def extract_gdf_labelled_data(gdf, id_str):
    """
    Extract label, confidence, URL, and labeller from manually labelled sample data

    Args:
        gdf (gpd.GeoDataFrame): manually labelled sample data
        id_str (str): name of building ID substring to search for in 'description' column of gdf

    Returns:
        gpd.GeoDataFrame: extracted information for manually labelled sample data
    """
    gdf[id_str] = gdf.description.str.extract(r"building_id: (.+) labeller")
    gdf["label"] = gdf.Name.str[:2]
    gdf["confidence"] = gdf["Name"].str[-1:]
    gdf["url"] = gdf.description.str.extract(r"Location: (.+) N")
    gdf["labeller"] = gdf.description.str.extract(r"labeller: (\w+)")
    return gdf


keep_cols = ["oct25_building_id", "label", "confidence", "url", "labeller"]

gdfs = []

for version, datasets in labelled_datasets.items():
    for labeller, file in datasets.items():
        print(labeller)
        gdf = gpd.read_file(file)
        if version == "apr25":
            gdf = extract_gdf_labelled_data(gdf, "apr25_building_id")
            print("Mapping apr25 IDs to oct25 IDs...")
            gdf["oct25_building_id"] = gdf["apr25_building_id"].map(apr_to_oct_id.get)
            gdfs.append(gdf[keep_cols])
        else:
            gdfs.append(extract_gdf_labelled_data(gdf, "oct25_building_id")[keep_cols])

all_labels_df = pl.from_pandas(pd.concat(gdfs))

# %% [markdown]
# ### Process samples which received labels from two labellers
#
# For the sake of the comparison between labels on the same building from multiple labellers, we currently separate out the "TT" category from the binary categorisation. This is because we are uncertain of how this should be classified in a binary given it represents a building footprint that contains both a block of flats and another built form.
#
# We group the nine labels into four: block; non-block; TT (contains block and non-block); and UL (unlabellable) to conduct the comparison between labellers. Because the labels will ultimately be used for a binary classification, we allow for disagreement between labels as long as they are within the same group of the four given.

# %%
double_labelled_df = (
    all_labels_df.with_columns(
        # Group labels into four categories: block, non-block, TT (mixed), UL (unlabellable)
        pl.when(pl.col("label").is_in(["BF", "BP", "SF"]))
        .then(pl.lit("block"))
        .when(pl.col("label").is_in(["CO", "TE", "TF", "OF", "WB"]))
        .then(pl.lit("non-block"))
        .when(pl.col("label") == "TT")
        .then(pl.lit("TT"))
        .otherwise(pl.lit("UL"))
        .alias("block_of_flats"),
        pl.col("oct25_building_id").is_duplicated().alias("duplicate"),
        # Filter to duplicated building IDs (meaning the building has been labelled by two different people)
    )
    .filter(pl.col("duplicate"))
    .group_by("oct25_building_id", maintain_order=True)
    .agg(
        # Aggregate the multiple labels and other information into a list per building
        pl.col("label"),
        pl.col("block_of_flats"),
        pl.col("confidence"),
        pl.col("labeller"),
        pl.col("url").first(),
    )
    .with_columns(
        # Convert lists to structs and then unnest to create one row per building with information from both labellers
        pl.col("label").list.to_struct(fields=["sublabel_0", "sublabel_1"]),
        pl.col("block_of_flats").list.to_struct(fields=["label_0", "label_1"]),
        pl.col("confidence").list.to_struct(fields=["confidence_0", "confidence_1"]),
        pl.col("labeller").list.to_struct(fields=["labeller_0", "labeller_1"]),
    )
    .unnest(columns=["label", "block_of_flats", "confidence", "labeller"])
    .with_columns(
        (pl.col("label_0") == pl.col("label_1")).alias("agree_label"),
        (pl.col("sublabel_0") == pl.col("sublabel_1")).alias("agree_sublabel"),
    )
)

# Building IDs where the labels (of the aggregated four categories) do not match between labellers
drop_building_ids = double_labelled_df.filter(~pl.col("agree_label"))[
    "oct25_building_id"
].unique()

# Building IDs where the labels (of the aggregated four categories) match between labellers
agree_buildings_ids = double_labelled_df.filter(pl.col("agree_label"))[
    "oct25_building_id"
].unique()


# Filter to final dataset of labelled data for model
final_labels_df = (
    all_labels_df.filter(
        # Drop buildings where labellers do not agree
        ~pl.col("oct25_building_id").is_in(drop_building_ids)
    )
    .cast({"confidence": pl.Int16}, strict=False)
    .with_columns(
        # For buildings where labellers agree, set the confidence to 1, otherwise retain original confidence
        pl.when(pl.col("oct25_building_id").is_in(agree_buildings_ids))
        .then(1)
        .otherwise(pl.col("confidence"))
        .alias("confidence")
        # Drop duplicate building IDs - note that this affects the distribution of subclasses within groups
    )
    .unique(subset=["oct25_building_id"], keep="any", maintain_order=True)
    .with_columns(
        # Convert labels to binary classification for model training
        pl.when(pl.col("label").is_in(block_of_flats_labels))
        .then(True)
        .when(pl.col("label").is_in(non_block_labels))
        .then(False)
        .otherwise(None)
        .alias("block_of_flats")
    )
)

# %% [markdown]
# ## Plot model feature distributions

# %%
# Join features to labelled data, keep rows where confidence of label is 1 and drop unlabellable rows
model_df = features_df.join(
    final_labels_df.select(
        ["oct25_building_id", "label", "confidence", "block_of_flats"]
    ),
    how="inner",
    on="oct25_building_id",
).filter(pl.col("confidence") == 1, pl.col("label") != "UL")

# Model features
features = [
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

labels = model_df["label"].unique()

# %%
model_df["block_of_flats"].value_counts(normalize=True)

# %%
model_df["label"].value_counts(normalize=True).sort("proportion", descending=True)

# %%
model_df["label"].value_counts().sort("count", descending=True)

# %%
# Plot typical spatial distribution of UPRNs within each building archetype
typical_examples = {}

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 7))

for ax, label in zip(axes.ravel(), labels):

    # Median area of concave hull
    median_concave_hull = model_df.filter(pl.col("label") == label)[
        "concave_hull_area_m2"
    ].median()

    # Get the building with a concave hull area which is closest to the median
    diff = (pl.col("concave_hull_area_m2") - median_concave_hull).abs()
    typical_building = model_df.filter(pl.col("label") == label).filter(
        diff.min() == diff
    )

    # Save building ID and number of flats for the selected building
    typical_examples[f"{label}_building_id"] = typical_building["oct25_building_id"][0]
    typical_examples[f"{label}_n_flats_for_selected_building"] = typical_building[
        "n_flats"
    ][0]

    # Get the median average number of flats, proportion of flats, and avg number of stacked UPRNs in each building type
    typical_examples[f"{label}_median_n_flats"] = model_df.filter(
        pl.col("label") == label,
    )["n_flats"].median()
    typical_examples[f"{label}_median_proportion_flats"] = model_df.filter(
        pl.col("label") == label,
    )["proportion_flats"].median()
    typical_examples[f"{label}_median_avg_n_stacked_uprns"] = model_df.filter(
        pl.col("label") == label,
    )["avg_n_stacked_uprns"].median()

    # Get the geometry of the example building
    building_id = typical_examples[f"{label}_building_id"]
    examples_gdf = buildings_w_uprns_gdf[
        buildings_w_uprns_gdf["oct25_building_id"] == building_id
    ][["geometry"]]
    examples_gdf["colour"] = "blue"

    # Get the UPRNs of the example building
    temp_examples_gdf = all_uprns_gdf[
        all_uprns_gdf["oct25_building_id"] == building_id
    ][["geometry"]]
    temp_examples_gdf["colour"] = "red"

    # Plot the above info
    examples_gdf = pd.concat([examples_gdf, temp_examples_gdf])
    examples_gdf.plot(ax=ax, color=examples_gdf["colour"])
    ax.set_title(
        f"{label}, N flats: {typical_examples[f'{label}_n_flats_for_selected_building']}"
    )

fig.suptitle(
    "Typical distribution of UPRNs in building footprint per archetype\n(based on median concave hull (m2))"
)
fig.tight_layout()

# %%
# Plot the distribution of features for blocks and non blocks
nrows = 3
ncols = math.ceil(len(features) / nrows)

fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8))

for ax, feature in zip(axs.ravel(), features):
    ax.hist(model_df.filter(pl.col("block_of_flats"))[feature], bins=40, alpha=0.5)
    ax.hist(model_df.filter(~pl.col("block_of_flats"))[feature], bins=40, alpha=0.5)
    ax.set_title(feature)

fig.legend(["Block of flats", "Non block"], loc="upper right")
fig.suptitle("Distribution of features in blocks and non-blocks")
fig.tight_layout()

# %%
# Same but with violin plots
nrows = 3
ncols = math.ceil(len(features) / nrows)

fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8))

for ax, feature in zip(axs.ravel(), features):
    vplot = ax.violinplot(
        dataset=(
            model_df.filter(pl.col("block_of_flats"))[feature],
            model_df.filter(~pl.col("block_of_flats"))[feature],
        )
    )
    for i, pc in enumerate(vplot["bodies"], 1):
        if i % 2 != 0:
            pc.set_facecolor("gold")
        else:
            pc.set_facecolor("royalblue")
        pc.set_edgecolor("black")

    ax.set_title(feature)

fig.legend(["Block of flats", "Non block"], loc="upper right")
fig.suptitle("Distribution of features in blocks and non-blocks")
fig.tight_layout()

# %% [markdown]
# ## Train and evaluate model and tune hyperparameters

# %%
# Set random state int and RandomState instance
random_state = 8
rng = np.random.RandomState(random_state)

# %%
# Model features
features = [
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

# %%
# Create param distributions for hyperparameter search
param_distributions = {
    "n_estimators": list(range(100, 1000, 10)),
    "max_depth": list(range(10, 100, 5)),
    "min_samples_split": list(range(2, 20, 1)),
    "min_samples_leaf": list(range(1, 10, 1)),
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"],
}

# Sort model dataframe so that results are replicable
pd_model_df = (
    model_df.to_pandas().set_index("oct25_building_id").sort_values("oct25_building_id")
)
X = pd_model_df[features]
y = pd_model_df["block_of_flats"]

# Keep a final validation set aside
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)

# Create cross-validation splitter and classifier
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
rfc = RandomForestClassifier(random_state=rng)

# Conduct halving random search for hyperparameters and cross-validation
search = HalvingRandomSearchCV(
    estimator=rfc,
    param_distributions=param_distributions,
    factor=2,  # Reduce halving aggressiveness
    random_state=random_state,
    cv=cv,  # Defaults to use StratifiedKFold splitter with 5 folds because model is binary classifier
    scoring="f1",  # Use F1 scoring metric for optimisation
    n_jobs=-1,  # Use all available cores
).fit(X_train, y_train)

print(f"Best F1 score: {search.best_score_}")
print(f"Best params: {search.best_params_}")

# %%
# Train final classifier model on full training set with the selected hyperparameters
final_model = RandomForestClassifier(**search.best_params_, random_state=rng)
final_model.fit(X_train, y_train)

# Evaluate final model on hold out validation set
y_pred = final_model.predict(X_val)
val_f1 = f1_score(y_val, y_pred)
print(f"F1 score on hold out validation set: {val_f1}")

# Plot confusion matrix results for hold out validation set
cmx = confusion_matrix(y_val, y_pred, normalize="true")
ConfusionMatrixDisplay(cmx).plot()
plt.title("Labels: 1=Block of flats, 0=Not")
plt.grid(False)
plt.show()

# %%
# Run model on buildings which weren't manually labelled
manually_labelled_ids = model_df["oct25_building_id"].unique()
to_label_df = features_df.filter(
    ~pl.col("oct25_building_id").is_in(manually_labelled_ids)
).to_pandas()
to_label_df["block_of_flats"] = final_model.predict(
    to_label_df.set_index("oct25_building_id")[features]
)
print(f'\n{to_label_df["block_of_flats"].value_counts(normalize=True)}\n')

# %% [markdown]
# ### Run additional model evaluation on final model

# %%
metrics = {
    "repeat": [],
    "f1_score": [],
    "precision_score": [],
    "recall_score": [],
    "accuracy": [],
    "roc_auc_score": [],
    "confusion_matrix": [],
}

# Use repeated stratified k-fold with 10 repeats for robustness
n_splits = 5
rskfolds = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=10, random_state=random_state
)
rskfolds.get_n_splits()

rfc = RandomForestClassifier(**search.best_params_, random_state=rng)

for i, (train_index, test_index) in enumerate(rskfolds.split(X_train, y_train)):
    # Get the repeat number
    if i % n_splits == 0:
        repeat = i / n_splits

    # Create train and test sets
    _X_train, _y_train = (
        X_train.loc[X_train.reset_index().index.isin(train_index)],
        y_train.loc[y_train.reset_index().index.isin(train_index)].values.ravel(),
    )
    _X_test, _y_test = (
        X_train.loc[X_train.reset_index().index.isin(test_index)],
        y_train.loc[y_train.reset_index().index.isin(test_index)].values.ravel(),
    )

    # Fit the classifier and predict classes
    rfc.fit(_X_train, _y_train)
    y_pred = rfc.predict(_X_test)
    y_pred_proba = rfc.predict_proba(_X_test)[:, 1]

    # Calculate metrics
    metrics["repeat"].append(repeat)
    metrics["f1_score"].append(f1_score(_y_test, y_pred))
    metrics["precision_score"].append(precision_score(_y_test, y_pred))
    metrics["recall_score"].append(recall_score(_y_test, y_pred))
    metrics["accuracy"].append(rfc.score(_X_test, _y_test))
    metrics["roc_auc_score"].append(roc_auc_score(_y_test, y_pred_proba))
    metrics["confusion_matrix"].append(
        confusion_matrix(_y_test, y_pred, normalize="true")
    )

# Calculate averages across all folds and repeats
for metric, v in metrics.items():
    if metric not in ["confusion_matrix", "repeat"]:
        print(f"Average {metric}: {np.mean(v)}")
        print(f"STD {metric}: {np.std(v)}\n")

# %%
# Calculate average metric per repeat
metrics_df = pl.DataFrame({k: v for k, v in metrics.items() if k != "confusion_matrix"})
repeats_df = metrics_df.group_by("repeat").agg(pl.all().mean()).sort(by="repeat")

repeats_df

# %%
# Calculate average metric across repeats
repeats_df.mean()

# %%
# View precision-recall curve
display = PrecisionRecallDisplay.from_predictions(
    _y_test, y_pred_proba, name="RF Classifier", plot_chance_level=True, despine=True
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
