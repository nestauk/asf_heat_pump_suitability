# %% [markdown]
# # Test different methodologies to fill building rise/ storey count information using building height
#
# FLAT_STOREY_COUNT data is available from EPC data and building height estimates are available from Microsoft ML Global Building Footprints dataset.
#
# FLAT_STOREY_COUNT has low coverage of non-null values in EPC and Microsoft ML Global Building Footprints has decent coverage. Here we test 4 different basic methodologies for converting building height data to FLAT STOREY COUNT data. We compare the methodologies at the end to select the best one.

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn import metrics
import statsmodels.formula.api as sm
from tqdm import tqdm
import geopandas as gdf
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.analysis.flats_on_fossils.features import building_rise

# %% [markdown]
# ## 1. Load processed EPC data for flats
# This file is the result of processing the EPC data with the `asf_heat_pump_suitability/analysis/flats_with_fossils/run_process_epc_flats.py` script.

# %%
# Import latest EPC data
flats_epc_path = "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/2024_Q3_epc_flats_processed.parquet"
raw_flats_epc_df = pl.read_parquet(flats_epc_path)

# %%
# This replaces values below 1 or any non-integer values with Nulls
flats_epc_df = building_rise.clean_col_flat_storey_count(raw_flats_epc_df)

# %%
raw_flats_epc_df["height"].describe()

# %%
flats_epc_df = (
    flats_epc_df.with_columns(pl.col("FLAT_STOREY_COUNT").cast(pl.Float64))
    .with_columns(
        pl.when(pl.col("height") >= 2)
        .then(pl.col("height"))
        .otherwise(None)
        .alias("height"),
        # Manual check shows the building with max storey count is incorrect
        pl.when(pl.col("UPRN") == "100080510780.0")
        .then(3)
        .otherwise(pl.col("FLAT_STOREY_COUNT"))
        .alias("FLAT_STOREY_COUNT"),
        raw_height=pl.col("height"),
    )
    .filter(pl.col("COUNTRY") != "Scotland")
)

# %% [markdown]
# ## 2. Add tall buildings in UK
# It appears that high rises likely don't have the correct height / storey count data (see `analysis_explore_storey_and_height_data.py` notebook) and the models are therefore underestimating their storey counts because the heights and storey counts are too low. We will try filling some of this data in using data about the tallest buildings in the UK taken from Wikipedia (URL: https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_the_United_Kingdom) and tall buildings in Manchester (URL: https://en.wikipedia.org/wiki/List_of_tallest_buildings_and_structures_in_Greater_Manchester). The dataset covers all current buildings which are over 100m tall in the UK and over 50m in Manchester. The data contains building max height in meters and storey count.
#
# We're interested in using the storey count information only, not height data, to improve our model. This is because our model converts height to storey counts and correcting only some, not all, of the height data will skew our model.
#
# We will use some basic string matching to join tall buildings to EPC data using building name and EPC address data.

# %%
# Load and preprocess tall buildings data for UK
tall_buildings_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/tallest_buildings_in_uk.csv",
    ignore_errors=True,
)

tall_buildings_df = (
    tall_buildings_df.drop_nulls(subset=["Official Name", "City", "Height (m)"])
    .select(["Official Name", "City", "County", "Borough", "Height (m)", "Floors"])
    .with_columns(pl.col("Floors").cast(pl.Int64))
    .rename(
        {
            "Official Name": "building_name",
        }
    )
)

# %%
# Load and preprocess tall buildings data for Manchester
manchester_tall_buildings_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/List of tallest buildings and structures in Greater Manchester.csv"
)

manchester_tall_buildings_df = (
    manchester_tall_buildings_df.drop_nulls(subset=["Name", "Height", "Location"])
    .select(["Name", "Height", "Floors", "Location"])
    .with_columns(
        pl.col("Floors").replace("N/A", None).cast(pl.Int64),
        pl.col("Height")
        .str.extract(r"\d+", group_index=0)
        .cast(pl.Float64)
        .alias("Height"),
        City=pl.lit("Manchester"),
        County=pl.lit("Greater Manchester"),
    )
    .rename({"Name": "building_name", "Location": "Borough", "Height": "Height (m)"})
    .select(["building_name", "City", "County", "Borough", "Height (m)", "Floors"])
)

# %%
# Concat the two tall building dfs
tall_buildings_df = pl.concat([tall_buildings_df, manchester_tall_buildings_df]).unique(
    subset=["building_name"]
)

# %%
# Get tall building names for string matching
tall_buildings_names = tall_buildings_df["building_name"].to_list()
tall_buildings_names = [
    building_name for building_name in tall_buildings_names if building_name
]

# %%
# Create full address from EPC data
tall_epc_buildings_df = flats_epc_df.with_columns(
    pl.concat_str(
        [pl.col("ADDRESS1"), pl.col("ADDRESS2"), pl.col("LOCAL_AUTHORITY_LABEL")],
        separator=" ",
    )
    .str.to_lowercase()
    .alias("full_address"),
    building_name=None,
)

# %%
# For each tall building, search the EPC address to see if it contains the building name
for building_name in tqdm(tall_buildings_names):
    tall_epc_buildings_df = tall_epc_buildings_df.with_columns(
        pl.when(pl.col("full_address").str.contains(building_name.lower()))
        .then(pl.lit(building_name))
        .otherwise(pl.col("building_name"))
        .alias("building_name")
    )

# %%
# Join the rest of the building data onto the EPC data where there is a match to a tall building
tall_epc_buildings_df = tall_epc_buildings_df.join(
    tall_buildings_df, how="left", on="building_name"
)
tall_epc_buildings_df = tall_epc_buildings_df.filter(
    pl.col("building_name").is_not_null()
)

# %%
# Filter to only accurate matches of EPC records to tall buildings
# A match is considered accurate if the full EPC address also contains the same borough / city / county information as the building
tall_epc_buildings_df = (
    tall_epc_buildings_df.with_columns(
        pl.when(
            (pl.col("full_address").str.contains(pl.col("Borough").str.to_lowercase()))
            | (pl.col("full_address").str.contains(pl.col("City").str.to_lowercase()))
            | (pl.col("full_address").str.contains(pl.col("County").str.to_lowercase()))
        )
        .then(True)
        .otherwise(False)
        .alias("accurate_match")
    )
    .filter(pl.col("accurate_match"))
    .select(["UPRN", "Floors", "Height (m)"])
)

# %%
tall_epc_buildings_df.shape

# %%
flats_epc_df = flats_epc_df.join(tall_epc_buildings_df, how="left", on="UPRN")

# %%
flats_epc_df = flats_epc_df.with_columns(
    (pl.col("Floors") - pl.col("FLAT_STOREY_COUNT")).alias("diff")
)

# %%
flats_epc_df.filter(pl.col("diff").is_not_null())["diff"].describe()

# %%
flats_epc_df = flats_epc_df.with_columns(
    pl.col("Floors").fill_null(pl.col("FLAT_STOREY_COUNT")).alias("FLAT_STOREY_COUNT")
)

# %%
# save_utils.save_to_s3(
#     flats_epc_df,
#     "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/2024_Q3_epc_flats_processed_with_tall_buildings_FOR_ANALYSIS.parquet",
# )

# %% [markdown]
# ## Method 1: use a single meters per storey value
# Use a single meters per storey value derived from government information about high-rises. Govt defines high rises as buildings which are at least 18m or 7 storeys.

# %%
mps = 18 / 7

# %%
flats_epc_df = flats_epc_df.with_columns(
    (pl.col("height") / mps).round().alias("method_1_storey_count")
)

# %%
flats_epc_df["method_1_storey_count"].describe()

# %% [markdown]
# ## Method 2: Building rise thresholds
# Set a threshold height (mean of upper quartile of lower rise and lower quartile of higher rise) for each building rise and then use median meters per storey for each rise type to calculate the number of storeys of each building from the height.
# In this methodology, we classify the building rise type first and then calculate the storey count.


# %%
def calculate_dict_rise_thresholds(df: pl.DataFrame) -> dict:
    """
    Calculate median meters per storey per building rise type, and thresholds for minimum height of mid-rises and high-rises.

    Args:
        df (pl.DataFrame): EPC flat records with `FLAT_STOREY_COUNT` and building `height` variables

    Returns:
        dict: meters per storey for each building rise type and threshold for min height of mid- and high-rises
    """
    df = df.filter(
        (pl.col("height").is_not_null()) & (pl.col("FLAT_STOREY_COUNT").is_not_null())
    ).with_columns(
        (pl.col("height") / pl.col("FLAT_STOREY_COUNT")).alias("meters_per_storey"),
        pl.when((pl.col("FLAT_STOREY_COUNT") > 0) & (pl.col("FLAT_STOREY_COUNT") <= 3))
        .then(pl.lit("low-rise"))
        .when((pl.col("FLAT_STOREY_COUNT") > 3) & (pl.col("FLAT_STOREY_COUNT") <= 10))
        .then(pl.lit("medium-rise"))
        .when(pl.col("FLAT_STOREY_COUNT") > 10)
        .then(pl.lit("high-rise"))
        .otherwise(None)
        .alias("building_rise"),
    )

    low_rise_uq = df.filter(pl.col("building_rise") == "low-rise")["height"].quantile(
        0.75
    )
    mid_rise_lq = df.filter(pl.col("building_rise") == "medium-rise")[
        "height"
    ].quantile(0.25)

    mid_rise_uq = df.filter(pl.col("building_rise") == "medium-rise")[
        "height"
    ].quantile(0.75)
    high_rise_lq = df.filter(pl.col("building_rise") == "high-rise")["height"].quantile(
        0.25
    )

    median_df = df.group_by("building_rise").agg(pl.col("meters_per_storey").median())
    thresholds = {
        "meters_per_storey": dict(
            zip(median_df["building_rise"], median_df["meters_per_storey"])
        )
    }

    thresholds.update(
        {
            "rise_thresholds": {
                "mid_rise_min": np.mean([low_rise_uq, mid_rise_lq]),
                "high_rise_min": np.mean([mid_rise_uq, high_rise_lq]),
            }
        }
    )

    return thresholds


# %%
def extend_df_building_rise_from_thresholds(
    df: pl.DataFrame, mid_rise_min: float, high_rise_min: float, mps: dict
) -> pl.DataFrame:
    """
    Add `building_rise` column to dataframe using `FLAT_STOREY_COUNT` and `height` columns. Buildings are partitioned
    into low- (<=3 storeys), medium- (4-10 storeys) and high-rise (>10 storeys).

    Args:
        df (pl.DataFrame): EPC records with `FLAT_STOREY_COUNT` and building `height` data
        mid_rise_min (float): minimum building height of medium rises in meters
        high_rise_min (float): minimum building height of high rises in meters
        mps (dict): meters per storey used to calculate storey counts from building height

    Returns:
        pl.DataFrame: EPC data with `building_rise` column
    """
    df = (
        df.with_columns(
            pl.when(pl.col("height") >= high_rise_min)
            .then(pl.lit("high-rise"))
            .when(pl.col("height") >= mid_rise_min)
            .then(pl.lit("medium-rise"))
            .when(pl.col("height").is_not_null())
            .then(pl.lit("low-rise"))
            .otherwise(None)
            .alias("method_2_building_rise")
        )
        .with_columns(
            pl.col("method_2_building_rise")
            .replace(mps)
            .cast(pl.Float64)
            .alias("method_2_meters_per_storey")
        )
        .with_columns(
            (pl.col("height") / pl.col("method_2_meters_per_storey"))
            .round()
            .alias("method_2_storey_count")
        )
    )
    return df


# %%
params_dict = calculate_dict_rise_thresholds(flats_epc_df)
params_dict

# %%
flats_epc_df = extend_df_building_rise_from_thresholds(
    flats_epc_df,
    mid_rise_min=params_dict["rise_thresholds"]["mid_rise_min"],
    high_rise_min=params_dict["rise_thresholds"]["high_rise_min"],
    mps=params_dict["meters_per_storey"],
)

# %%
for rise in ["low-rise", "medium-rise", "high-rise"]:
    _rise_thresholds_df = flats_epc_df.filter(
        pl.col("FLAT_STOREY_COUNT").is_null(), pl.col("method_2_building_rise") == rise
    )
    print(_rise_thresholds_df["method_2_storey_count"].min())
    print(_rise_thresholds_df["method_2_storey_count"].max())
    print(_rise_thresholds_df["height"].max())

# %%
flats_epc_df["method_2_building_rise"].value_counts()

# %% [markdown]
# ## Method 3: linear regression
# Create a simple linear regression model with 2 features: building height and UPRNs per building, to predict flat storey count.
#
# First we prepare the dataset for modelling:
# - exclude rows with null values in features or target values
# - exclude rows where country is Scotland (due to systemic error discovered in Scottish data)
# - exclude rows where building height is <3m
#
# We don't set a minimum threshold for meters per storey because our hypothesis is that the building height data underestimates the height of taller buildings. Therefore, we are allowing meters per storey values which are smaller than is realistic to train the model to predict higher storey counts for taller buildings.
#
# We group buildings by building ID to ensure that the same buildings are not represented in both the training and test sets.

# %%
# Prepare model data subset
model_df = (
    flats_epc_df.filter(
        pl.col("FLAT_STOREY_COUNT").is_not_null(),
        pl.col("height").is_not_null(),
        pl.col("property_per_m2").is_not_null(),
    )
    .with_columns(
        (pl.col("height") / pl.col("FLAT_STOREY_COUNT")).alias("meters_per_storey")
    )
    .filter(pl.col("height") >= 2)
)

# %%
uprn_sample = model_df["UPRN"].to_list()
X = model_df.select(["height", "property_per_m2"])
y = model_df["FLAT_STOREY_COUNT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# %%
# Train model
reg = sklearn.linear_model.LinearRegression().fit(X_train, y_train)

# %%
# R2 on training data
reg.score(X_train, y_train)

# %%
# R2 on test data
reg.score(X_test, y_test)

# %%
reg.coef_

# %%
# Plot building height vs storey count for our whole train and test dataset
plt.scatter(
    X_train["height"], y_train, s=4, alpha=0.2, label="True storey count", marker="."
)
plt.scatter(
    X_train["height"],
    reg.predict(X_train),
    color="r",
    s=4,
    alpha=0.2,
    label="Predicted storey count",
    marker="v",
)
plt.title("Storey count versus estimated building height (m)")
plt.xlabel("Estimated building height (m)")
plt.ylabel("Storey count")
plt.legend()
plt.show()

# %%
# Predict storey count for all buildings with height data in the dataset
prediction_df = flats_epc_df.filter(
    pl.col("height").is_not_null(), pl.col("property_per_m2").is_not_null()
).select(["UPRN", "height", "property_per_m2"])

prediction_df = prediction_df.with_columns(
    method_3_storey_count=np.round(
        reg.predict(prediction_df.select(["height", "property_per_m2"]))
    )
)

# %%
# Join predictions to EPC flats dataset
if "method_3_storey_count" in flats_epc_df.columns:
    flats_epc_df = flats_epc_df.drop("method_3_storey_count")
flats_epc_df = flats_epc_df.join(
    prediction_df.select(["UPRN", "method_3_storey_count"]), how="left", on="UPRN"
)

# %% [markdown]
# ## Method 4: polynomial regression
# Here we conduct similar regression, but with polynomial features. This is because our hypothesis is that the height data underestimates height of taller buildings, therefore a polynomial equation may be more appropriate than linear.

# %%
poly_features = sklearn.preprocessing.PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

# %%
reg = sklearn.linear_model.LinearRegression().fit(X_poly, y)

# %%
# R2 value of model subset
reg.score(X_poly, y)

# %%
reg.intercept_

# %%
plt.scatter(model_df["height"], y, s=2, label="True storey count")
plt.scatter(
    model_df["height"],
    reg.predict(X_poly),
    color="r",
    s=1,
    label="Predicted storey count",
)
plt.ylim(0, 90)
plt.title("Storey count vs building height (m)")
plt.xlabel("Building height (m)")
plt.ylabel("Storey count")
plt.legend()
plt.show()

# %%
# Predict storey count for all buildings with height data in the dataset
prediction_df = flats_epc_df.filter(
    pl.col("height").is_not_null(), pl.col("property_per_m2").is_not_null()
).select(["UPRN", "height", "property_per_m2"])
prediction_df = prediction_df.with_columns(
    method_4_storey_count=np.round(
        reg.predict(
            poly_features.fit_transform(
                prediction_df.select(["height", "property_per_m2"])
            )
        )
    )
)

# Join predictions to EPC flats dataset
if "method_4_storey_count" in flats_epc_df.columns:
    flats_epc_df = flats_epc_df.drop("method_4_storey_count")
flats_epc_df = flats_epc_df.join(
    prediction_df.select(["UPRN", "method_4_storey_count"]), how="left", on="UPRN"
)

# %% [markdown]
# ## Comparing methods
#
# Here we conduct some basic evaluation of our 4 models to determine which is the best to take forward.

# %%
# Filter to the sample of UPRNs we used for our regression models
evaluation_df = flats_epc_df.select(
    [
        "UPRN",
        "FLAT_STOREY_COUNT",
        "method_1_storey_count",
        "method_2_storey_count",
        "method_3_storey_count",
        "method_4_storey_count",
    ]
).filter(pl.col("UPRN").is_in(uprn_sample))

# %%
assert len(evaluation_df) == len(model_df) == len(uprn_sample)

# %%
# Summary statistics
evaluation_df.describe()

# %%
# Output R2, MAE, MSE per model
for method in [1, 2, 3, 4]:
    r2 = metrics.r2_score(
        y_true=evaluation_df["FLAT_STOREY_COUNT"],
        y_pred=evaluation_df[f"method_{method}_storey_count"],
    )
    mae = metrics.mean_absolute_error(
        y_true=evaluation_df["FLAT_STOREY_COUNT"],
        y_pred=evaluation_df[f"method_{method}_storey_count"],
    )
    mse = metrics.mean_squared_error(
        y_true=evaluation_df["FLAT_STOREY_COUNT"],
        y_pred=evaluation_df[f"method_{method}_storey_count"],
    )
    print(f"R2 score for method {method}: {r2}")
    print(f"Mean absolute error for method {method}: {mae}")
    print(f"Mean squared error for method {method}: {mse}\n")

# %%
# See distribution of flat storey count for each method
fig, axs = plt.subplots(1, 4, figsize=(15, 4), sharey=True)

for ax, method in zip(axs.ravel(), [1, 2, 3, 4]):
    plot_df = flats_epc_df.filter(pl.col(f"method_{method}_storey_count").is_not_null())
    plot_df = plot_df.filter(
        pl.col(f"method_{method}_storey_count") < 250,
        pl.col(f"method_{method}_storey_count") > 0,
    )
    ax.boxplot(plot_df[f"method_{method}_storey_count"])
    ax.set_title(f"Method {method}, N={len(plot_df)}")
    ax.set_ylabel("Storey count")

plt.suptitle(
    "Distribution of flat storey count in EPC flats using different methods of calculation"
)
plt.tight_layout()
plt.show()

# %%
# See number of EPC flats per storey count with each method
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

for ax, method in zip(axs.ravel(), [1, 2, 3, 4]):
    _temp_df = flats_epc_df.filter(
        pl.col(f"method_{method}_storey_count").is_not_null()
    ).select(["UPRN", f"method_{method}_storey_count", "height"])
    plot_df = _temp_df.group_by(f"method_{method}_storey_count").agg(
        count=pl.col("UPRN").count()
    )
    ax.bar(plot_df[f"method_{method}_storey_count"], plot_df["count"])
    ax.set_title(f"Method {method}, N={len(_temp_df)}")
    ax.set_xlabel("Storey count")
    ax.set_ylabel("Count of EPC records")
    ax.set_xlim(0, 50)

plt.suptitle(f"Number of EPC flats records per storey count")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Model selection: method 3, linear regression
#
# We will select method 3: linear regression with building height and UPRN density per building to estimate storey count from our height data. Method 4 has a slightly higher R2 value, but has predicted some extremely negative storey counts and is less explainable as a model.
#
# Below we do some additional exploration and validation of the results of method 3.

# %%
# Prepare model data subset
model_df = (
    flats_epc_df.filter(
        pl.col("FLAT_STOREY_COUNT").is_not_null(),
        pl.col("height").is_not_null(),
        pl.col("property_per_m2").is_not_null(),
    )
    .with_columns(
        (pl.col("height") / pl.col("FLAT_STOREY_COUNT")).alias("meters_per_storey")
    )
    .filter((pl.col("COUNTRY") != "Scotland"), (pl.col("height") >= 2))
)

# %%
model_df.shape

# %%
# Check feature correlation
np.corrcoef(X["height"], X["property_per_m2"])

# %% [markdown]
# ### Test LR on full analytical sample using `statsmodels`
#
# We use `statsmodels` so that we can see the confidence intervals of the coefficients.

# %%
uprn_sample = model_df["UPRN"].to_list()
X = model_df.select(["height", "property_per_m2"])
y = model_df["FLAT_STOREY_COUNT"]

# %%
model = sm.ols("FLAT_STOREY_COUNT ~ height + property_per_m2", model_df.to_pandas())
results = model.fit()

# %%
results.summary()

# %%
results.params

# %%
results.conf_int()

# %%
results.rsquared

# %% [markdown]
# #### Cross-validation with group ID
#
# We use building ID as group ID to ensure different flats in the same building are split into the same fold.
#
# Below, we split the model into 3 random folds 50 times and then iterate through training/testing the model on the folds. We take the average r2 values on the test sets.
#
# We can see that the R2 is variable. Deeper investigation (not shown in this notebook) has revealed that this is likely due to the presence of single outlier buildings in the test set that have a lot of flats in them and thus significantly affect the R2 value. E.g. we found a test set with an R2 of -0.10. After removal of a significant outlier building which contained 115 flats in the sample, the R2 increased to ~0.5.
#
# Therefore we use the combined knowledge of the coefficient confidence intervals and the average R2 to satisfy that the model is acceptably robust.

# %%
groups = model_df["building_id"]
gss = GroupShuffleSplit(n_splits=3, test_size=0.25)

# %%
avg_test_r2 = []

for i in tqdm(range(0, 50)):
    test_r2 = []
    for train_index, test_index in gss.split(X, y, groups):
        train_groups, test_groups = groups[train_index], groups[test_index]
        # Check the train and test sets are mutually exclusive
        assert not set(train_groups) & set(test_groups)

        X_train = model_df.filter(pl.col("building_id").is_in(train_groups)).select(
            ["height", "property_per_m2"]
        )
        X_test = model_df.filter(pl.col("building_id").is_in(test_groups)).select(
            ["height", "property_per_m2"]
        )

        y_train = model_df.filter(pl.col("building_id").is_in(train_groups))[
            "FLAT_STOREY_COUNT"
        ]
        y_test = model_df.filter(pl.col("building_id").is_in(test_groups))[
            "FLAT_STOREY_COUNT"
        ]

        # Train model
        reg = linear_model.LinearRegression().fit(X_train, y_train)

        # R2 on test data
        test_r2.append(reg.score(X_test, y_test))

    avg_test_r2.append(np.mean(test_r2))

print(avg_test_r2)
print(np.mean(avg_test_r2))
print(np.median(avg_test_r2))

# %% [markdown]
# ## Train and evaluate final selected model
#
# We train the final model on the full dataset. We're not too worried about overfitting because it's a linear model with 2 features.
#
# In the evaluation below we classify our actual and predicted storey counts into low-, medium-, and high-rise buildings and compare them. We can see that we pretty accurately predict the number of high-rises, we under-predict the number of low-rises, and over-predict the number of high-rises. We can see that our model accuracy is around 80%.
#
# Comparatively, the polynomial regression model has a much more accurate prediction of the number of medium-, and low-rises. However, the model accuracy is still around 80% so for our use case where we care about the accuracy of individual predictions, it's not necessarily a more useful model. It's also much less intuitive and explainable, especially when viewing the coefficients.
#
# When viewing our evaluation results, we should be cognizant of the fact that our ground truth y values have known errors. In the random sample of 15 flats we took, we found 3/15 had erroneous flat storey count data when compared to google maps, equivalent to 80%. So although our model shows inaccurate predictions for 20% of buildings in our labelled dataset, it could be the case that some of these labels are incorrect.

# %%
# Prepare model data subset
model_df = (
    flats_epc_df.filter(
        pl.col("FLAT_STOREY_COUNT").is_not_null(),
        pl.col("height").is_not_null(),
        pl.col("property_per_m2").is_not_null(),
    )
    .with_columns(
        (pl.col("height") / pl.col("FLAT_STOREY_COUNT")).alias("meters_per_storey")
    )
    .filter((pl.col("COUNTRY") != "Scotland"), (pl.col("height") >= 2))
)

uprn_sample = model_df["UPRN"].to_list()
X = model_df.select(["height", "property_per_m2"])
y = model_df["FLAT_STOREY_COUNT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# Train final candidate model
reg = linear_model.LinearRegression().fit(X, y)

# %%
reg.score(X, y)

# %%
# Get full data predictions of final candidate model
candidate_preds = reg.predict(X)
residuals = y - candidate_preds

plt.hist(residuals, bins=100)
plt.title(
    "Distribution of residuals from selected model for predicting flat storey count"
)
plt.show()

# %%
# Create dataframe of counts of flats in each rise class
pred_lists_to_plot = [y] + [candidate_preds]

# # Uncomment to see the results for the polynomial linear regression model
# poly_reg = sklearn.linear_model.LinearRegression().fit(X_poly, y)
# pred_lists_to_plot = [y] + [poly_reg.predict(X_poly)]

labels = {
    "preds_0": "Actual",
    "preds_1": "Candidate model",
}

results_dfs = []
class_counts_dfs = []

for i, predictions in enumerate(pred_lists_to_plot):
    results_df = pl.DataFrame(
        pl.Series(f"storeys", values=np.round(predictions))
    ).with_columns(
        pl.when(pl.col("storeys") >= 10)
        .then(pl.lit("High-rise"))
        .when(pl.col("storeys") <= 3)
        .then(pl.lit("Low-rise"))
        .otherwise(pl.lit("Medium-rise"))
        .alias("rise_class")
    )
    results_dfs.append(results_df.rename({"rise_class": f"preds_{i}"}))
    class_counts_dfs.append(
        results_df["rise_class"].value_counts().rename({"count": f"preds_{i}"})
    )

results_df = (
    pl.DataFrame(class_counts_dfs[0])
    .join(pl.DataFrame(class_counts_dfs[1]), how="outer", on="rise_class")
    .with_columns(
        pl.coalesce(pl.col(["rise_class", "rise_class_right"])).alias("storeys")
    )
    .drop(pl.col("rise_class_right"))
    .rename(labels)
)

# --------------------------------------------------------------- #
# Plot bar chart of actual vs predicted classes
custom_sort_key = {"Low-rise": 0, "Medium-rise": 1, "High-rise": 2}

ax = (
    results_df.to_pandas()
    .sort_values("rise_class", key=lambda x: x.map(custom_sort_key))
    .set_index("rise_class")
    .plot(kind="bar", figsize=(10, 5))
)
for axc in ax.containers:
    ax.bar_label(axc)
plt.xlabel("Rise class count")
plt.ylabel("Count of flats")
plt.title(
    "Rise class per predicted storey count (after rounding) from candidate model and actuals"
)
plt.show()


# --------------------------------------------------------------- #
# Calculate proportions of classes
for col in ["Actual", "Candidate model"]:
    _df = results_df.with_columns(
        pl.col(col).sum().alias(f"{col}_total"),
    ).with_columns((pl.col(col) / pl.col(f"{col}_total")).alias(f"{col}_proportion"))

print(_df)


# --------------------------------------------------------------- #
# Calculate accurate predictions
prediction_accuracy_df = (
    pl.concat([df.drop("storeys") for df in results_dfs], how="horizontal")
    .rename(labels)
    .with_columns(
        pl.when(pl.col("Actual") != pl.col("Candidate model"))
        .then(False)
        .otherwise(True)
        .alias("accurate_prediction")
    )
)

print(prediction_accuracy_df["accurate_prediction"].value_counts())
print(prediction_accuracy_df["accurate_prediction"].value_counts(normalize=True))

# %%
pred_lists_to_plot = [y] + [candidate_preds]

labels = {
    "preds_0": "Actual",
    "preds_1": "Candidate model",
}

storey_counts_list = []
for i, predictions in enumerate(pred_lists_to_plot):
    storey_counts_list.append(
        pl.Series(f"storeys", values=np.round(predictions))
        .value_counts()
        .rename({"count": f"preds_{i}"})
    )

results_df = pl.DataFrame(storey_counts_list[0])
for l in storey_counts_list[1:]:
    results_df = (
        results_df.join(pl.DataFrame(l), how="outer", on="storeys")
        .with_columns(storeys=pl.coalesce(pl.col(["storeys", "storeys_right"])))
        .drop(pl.col("storeys_right"))
    )

results_df = results_df.rename(labels).sort(pl.col("storeys"))

results_df.to_pandas().set_index("storeys").plot(kind="bar", figsize=(10, 5))
plt.xlabel("Storey count")
plt.ylabel("Count of flats")
plt.title(
    "Count of flats per predicted storey count (after rounding) from candidate model, and models trained across 3 folds"
)
plt.xlim(0, 36)
plt.show()

# %% [markdown]
# ## Predict storey count
#
# Use the selected model to predict the storey counts on the full EPC flats data set

# %%
# Exclude storey counts below 1 and over 100 and fill missing FLAT STOREY COUNT data with these estimates
flats_epc_df = flats_epc_df.with_columns(
    pl.when(
        (pl.col("method_3_storey_count") < 1) | (pl.col("method_3_storey_count") > 100)
    )
    .then(None)
    .otherwise(pl.col("method_3_storey_count"))
    .alias("method_3_storey_count_clean")
).with_columns(
    pl.col("FLAT_STOREY_COUNT")
    .fill_null(pl.col("method_3_storey_count_clean"))
    .alias("storey_count")
)

# %%
# Classify building rise type
flats_epc_df = building_rise.extend_df_building_rise_from_storey_count(
    flats_epc_df, storey_col="storey_count", building_rise_col="building_rise"
)

# %%
# Check proportion of null values after filling missing data (it was 87% before)
print(flats_epc_df["storey_count"].is_null().sum() / len(flats_epc_df))

# %%
# Check storey count data
flats_epc_df["storey_count"].describe()

# %%
# Check building rise counts
flats_epc_df["building_rise"].value_counts()

# %%
# Compare with English Housing Survey Data

# Add a column with government definition of high-rise building (18m+ or 7 storeys, whichever comes first) to compare proportions to English Housing Survey 2023-2024
flats_epc_df = flats_epc_df.with_columns(
    pl.when(pl.col("storey_count").is_null() & pl.col("height").is_null())
    .then(None)
    .when((pl.col("storey_count") >= 7) | (pl.col("height") >= 18))
    .then(True)
    .otherwise(False)
    .alias("govt_defined_high_rise")
)


print(
    flats_epc_df.filter(pl.col("govt_defined_high_rise").is_not_null())[
        "govt_defined_high_rise"
    ].value_counts()
)


print(
    flats_epc_df.filter(
        pl.col("govt_defined_high_rise").is_not_null(), pl.col("COUNTRY") == "England"
    )["govt_defined_high_rise"].value_counts(normalize=True)
)


# EHS reports 565,163 purpose-build high-rise flats in 2023 out of a total of 5,456,378 flats
# (Total is determined by adding together 'converted flat', 'purpose built flat, low rise', and 'purpose built flat, high rise' values).
# We can see our proportion compares quite well to the EHS proportion
565163 / (949600 + 3941615 + 565163)

# %%
save_utils.save_to_s3(
    flats_epc_df,
    "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/2024_Q3_epc_flats_processed_filled_storey_count.parquet",
)

# %%
