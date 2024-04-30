# %%
import random
import polars as pl
import balance

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters

# %% [markdown]
# ## Import sample

# %%
sample_path = "s3://asf-heat-pump-suitability/outputs/epc_sample_lsoa.parquet"

# %%
sample = pl.read_parquet(sample_path)

# %%
sample["lsoa"].value_counts().sort(by="count").to_dicts()

# %% [markdown]
# ## Prepare sample

# %%
# Filter out NA values
sample = sample.filter(~pl.col("BUILT_FORM").is_in(["", "unknown"]))
sample = sample.filter(~pl.col("TENURE").is_in(["", "unknown"]))

# %%
# Convert property_type categories to match census data
sample = sample.with_columns(
    (
        (pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]))
        & (pl.col("BUILT_FORM") == "Detached")
    ).alias("Detached whole house or bungalow")
)

sample = sample.with_columns(
    (
        (pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]))
        & (pl.col("BUILT_FORM") == "Semi-Detached")
    ).alias("Semi-detached whole house or bungalow")
)

terraced = [
    "Mid-Terrace",
    "End-Terrace",
    "Enclosed Mid-Terrace",
    "Enclosed End-Terrace",
]

sample = sample.with_columns(
    (
        (pl.col("PROPERTY_TYPE").is_in(["House", "Bungalow"]))
        & (pl.col("BUILT_FORM").is_in(terraced))
    ).alias("Terraced (including end-terrace) whole house or bungalow")
)

sample = sample.with_columns(
    (pl.col("PROPERTY_TYPE").is_in(["Flat", "Maisonette"])).alias(
        "Flat, maisonette or apartment"
    )
)
sample = sample.with_columns(
    (pl.col("PROPERTY_TYPE").is_in(["Park Home"])).alias(
        "A caravan or other mobile or temporary structure"
    )
)

# %%
# Transpose property_type data to same form as target
_cols = [
    "Detached whole house or bungalow",
    "Semi-detached whole house or bungalow",
    "Terraced (including end-terrace) whole house or bungalow",
    "Flat, maisonette or apartment",
]

sample_T = (
    sample.melt(id_vars="UPRN", value_vars=_cols)
    .filter(pl.col("value"))
    .rename({"variable": "property_type"})
    .select(pl.col(["UPRN", "property_type"]))
)

# %%
# Select relevant cols from sample
sample = (
    sample.join(sample_T, how="inner", on="UPRN")
    .select(pl.col(["lsoa", "UPRN", "TENURE", "property_type"]))
    .rename({"TENURE": "tenure", "UPRN": "id"})
)

# %% [markdown]
# ## Import validation sets

# %%
# Load property_type data from census
property_type_path = config["data_source"]["EW_housing_characteristics_target_url"]
property_type = base_getters.get_df_from_excel_url(
    property_type_path, sheet_name="2c", engine="calamine"
)

property_type = property_type.rename(
    property_type[2].to_dicts().pop()
)  # rename cols with line of df containing data
property_type = property_type.slice(
    3,
)  # remove null rows
property_type = property_type.rename({"Area Code": "lsoa"})

# %%
# Load tenure data from census
tenure_path = config["data_source"]["EW_housing_characteristics_target_url"]
tenure = base_getters.get_df_from_excel_url(
    tenure_path, sheet_name="3c", engine="calamine"
)

tenure = tenure.rename(tenure[2].to_dicts().pop())
tenure = tenure.slice(
    3,
)
tenure = tenure.rename(
    {
        "Area Code": "lsoa",
        "Owned or shared ownership": "owner-occupied",
        "Social Rented": "rental (social)",
        "Private Rented or lives rent free": "rental (private)",
    }
)

# %%
# Replace censored values ("c") with 0 value and convert cols to int
censored_vals = 0

int_cols = [
    "Detached whole house or bungalow",
    "Semi-detached whole house or bungalow",
    "Terraced (including end-terrace) whole house or bungalow",
    "Flat, maisonette or apartment",
    "A caravan or other mobile or temporary structure",
]

property_type = property_type.with_columns(
    [pl.col(int_cols).str.replace("c", f"{censored_vals}").cast(pl.Int64)]
).select(
    [
        "lsoa",
        "Detached whole house or bungalow",
        "Semi-detached whole house or bungalow",
        "Terraced (including end-terrace) whole house or bungalow",
        "Flat, maisonette or apartment",
        "A caravan or other mobile or temporary structure",
    ]
)

int_cols = ["owner-occupied", "rental (private)", "rental (social)"]
tenure = tenure.with_columns(
    [pl.col(int_cols).str.replace("c", f"{censored_vals}").cast(pl.Int64)]
).select(pl.col(["lsoa", "owner-occupied", "rental (private)", "rental (social)"]))

# %% [markdown]
# ## Get marginals

# %%
# Choose test LSOA
test_lsoa = "E01002058"  # small sample
# test_lsoa = "W01000527" # good example
# test_lsoa = "E01012376"  # missing val
# test_lsoa = "W01000328"  # example in slides
# test_lsoa = "E01033942"  # example on slack

# %%
# Proportions of tenure in census data
tenure_subset = tenure.filter(pl.col("lsoa") == test_lsoa)
tenure_dict = tenure_subset.to_dicts()[0]
del tenure_dict["lsoa"]

# %%
# Proportions of property_type in census data
property_type_subset = property_type.filter(pl.col("lsoa") == test_lsoa)
property_dict = property_type_subset.to_dicts()[0]
del property_dict["lsoa"]

# %%
# Generate artificial target population dataset with proportions derived from census data
marginals = {
    "tenure": {k: v / sum(tenure_dict.values()) for k, v in tenure_dict.items()},
    "property_type": {
        k: v / sum(property_dict.values()) for k, v in property_dict.items()
    },
}
target_df = balance.weighting_methods.rake.prepare_marginal_dist_for_raking(marginals)
target = balance.Sample.from_frame(target_df)

# %% [markdown]
# ## Add dummy rows to sample subset

# %%
subset = sample.filter(pl.col("lsoa") == test_lsoa)

# %%
target_features = {"tenure": tenure_dict, "property_type": property_dict}


# %%
def generate_df_dummies(feature_dicts):
    # Identify categories present in target but not sample
    dummies = []
    for feature, _dict in feature_dicts.items():
        _relevant_cats = {
            k for k in _dict.keys() if _dict.get(k) > 0
        }  # only add dummy rows for missing categories where target has observations but sample doesn't
        _missing_cats = _relevant_cats.difference(set(subset[feature].unique()))
        dummies.append({feature: list(_missing_cats)})
    return [pl.DataFrame(d) for d in dummies]


# %%
_dfs = generate_df_dummies(target_features)

dummy_rows = pl.concat(_dfs, how="horizontal").with_columns(
    pl.lit(test_lsoa, pl.String).alias("lsoa")
)
dummy_rows = dummy_rows.with_columns(
    pl.Series(name="id", values=[f"dummy_{_}" for _ in range(0, len(dummy_rows))])
)  # create dummy df


# Fill in null values in dummy df
for feature in ["tenure", "property_type"]:
    missing_vals = list(dummy_rows.drop_nulls()[feature].unique())
    if dummy_rows[feature].is_null().sum() > 0:
        if len(missing_vals) > 0:  # fill with random missing category first
            dummy_rows = dummy_rows.with_columns(
                pl.col(feature).fill_null(random.choices(missing_vals, k=1)[0])
            )
        else:  # otherwise, fill with category with max count in sample
            dummy_rows = dummy_rows.with_columns(
                pl.col(feature).fill_null(
                    subset[feature].value_counts().max()[feature][0]
                )
            )

# %%
subset = pl.concat([subset, dummy_rows[subset.columns]])

# %% [markdown]
# ## Reweighting sample

# %%
balance_sample = subset.to_pandas()
balance_sample = balance.Sample.from_frame(
    balance_sample[["id", "tenure", "property_type"]], id_column="id"
)
sample_w_target = balance_sample.set_target(target)
adjusted_sample = sample_w_target.adjust(method="rake")
assert adjusted_sample.is_adjusted()

# %% [markdown]
# ## Results

# %%
print(adjusted_sample.summary())

# %%
adjusted_sample.covars().plot()

# %%
weights = adjusted_sample.df
weights["weight_p"] = weights["weight"] / weights["weight"].sum()
weights
